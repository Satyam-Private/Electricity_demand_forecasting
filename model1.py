import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_dataset(data, look_back=24):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

def train_lstm_model(df, look_back=24, epochs=2, batch_size=32):
    # Use only the AEP_MW column for training
    data = df[['AEP_MW']].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Create training data
    X, y = create_dataset(data_scaled, look_back)
    
    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into train and test (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test),
              epochs=epochs, 
              batch_size=batch_size, 
              callbacks=[early_stop],
              verbose=1)
    
    return model, scaler

def make_predictions(model, scaler, df, hours_to_predict, look_back=24):
    # Use only the AEP_MW column
    data = df[['AEP_MW']].values
    
    # Normalize the data
    data_scaled = scaler.transform(data)
    
    # Initialize predictions array
    predictions = []
    
    # Take the last look_back hours as initial input
    current_input = data_scaled[-look_back:].reshape(1, look_back, 1)
    
    for _ in range(hours_to_predict):
        # Predict next hour
        current_pred = model.predict(current_input)
        predictions.append(current_pred[0, 0])
        
        # Update input for next prediction
        current_input = np.append(current_input[:, 1:, :], current_pred.reshape(1, 1, 1), axis=1)
    
    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions