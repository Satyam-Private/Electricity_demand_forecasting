from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model1 import train_lstm_model, make_predictions
import os
import json
app = Flask(__name__)

# Global variables to store data and model
global_df = None
global_model = None
global_scaler = None
@app.route('/')
def home():
    return render_template('landing.html')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            # Read and preprocess the data
            df = pd.read_csv(file)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            
            # Store the dataframe globally
            global global_df
            global_df = df
            
            # Train the model
            global global_model, global_scaler
            global_model, global_scaler = train_lstm_model(df)
            
            return redirect(url_for('forecast'))
    
    return render_template('index.html')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        if global_df is None or global_model is None:
            return redirect(url_for('index'))
        
        hours = int(request.form['hours'])
        
        # Make predictions
        predictions = make_predictions(global_model, global_scaler, global_df, hours)
        
        # Generate future timestamps
        last_timestamp = global_df.index[-1]
        future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, hours+1)]
        
        # Prepare data for visualization
        history = global_df['AEP_MW'].tail(24*7).reset_index()  # Last week of data
        history.columns = ['Datetime', 'AEP_MW']
        
        forecast_data = pd.DataFrame({
            'Datetime': future_timestamps,
            'AEP_MW': predictions.flatten()
        })
        
        return render_template('results.html', 
                             history=history.to_dict('records'),
                             forecast=forecast_data.to_dict('records'),
                             hours=hours)
    
    return render_template('forecast.html')



@app.route('/analyze-upload', methods=['GET', 'POST'])
def analyze_upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            return redirect(url_for('analyze', filename=file.filename))  # Optional: use session or save to disk
    return render_template('uploads.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(url_for('analyze_upload'))

    file = request.files['file']
    df = pd.read_csv(file)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Date'] = df['Datetime'].dt.date

    if 'solar_power' not in df.columns:
        df['solar_power'] = 0

    daily_demand = df.groupby('Date')['AEP_MW'].sum().reset_index()
    daily_solar = df.groupby('Date')['solar_power'].sum().reset_index()

    daily_summary = pd.merge(daily_demand, daily_solar, on='Date')
    daily_summary['Solar_Coverage'] = (daily_summary['solar_power'] / daily_summary['AEP_MW']) * 100

    peak_hour = df.loc[df['AEP_MW'].idxmax()]

    data = {
        'dates': daily_summary['Date'].astype(str).tolist(),
        'daily_demand': daily_summary['AEP_MW'].astype(float).tolist(),
        'daily_solar': daily_summary['solar_power'].astype(float).tolist(),
        'solar_coverage': daily_summary['Solar_Coverage'].astype(float).tolist(),
        'peak_hour': int(peak_hour['Hour']),
        'peak_demand': float(peak_hour['AEP_MW']),
        'peak_time': peak_hour['Datetime'].strftime('%Y-%m-%d %H:%M')
    }

    
    return render_template("analyze.html", data=data)




if __name__ == '__main__':
    app.run(debug=True)