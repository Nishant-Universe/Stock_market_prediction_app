# utils.py
import yfinance as yf
import pandas as pd
import os

def get_stock_data(ticker, start_date='2010-01-01', end_date=None, force_download=False):
    """
    Fetch stock data from Yahoo Finance or load from cache
    """
    os.makedirs('data', exist_ok=True)
    file_path = f'data/{ticker}.csv'
    
    if not force_download and os.path.exists(file_path):
        # Load from cache - ensure proper date handling
        data = pd.read_csv(file_path)
        if 'Date' in data.columns:
            data = data.set_index('Date')
        data.index = pd.to_datetime(data.index)  # Ensure index is datetime
    else:
        # Download from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(file_path)
    
    return data

# utils.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, lookback=60):
    """
    Preprocess data for LSTM training
    """
    # Use only the 'Close' price
    close_prices = data[['Close']].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create training data
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def train_test_split(X, y, test_size=0.2):
    """
    Split data into training and testing sets
    """
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

# utils.py
from tensorflow.keras.models import load_model
import joblib
import numpy as np

def predict_future_prices(ticker, data, days_to_predict=30):
    """
    Predict future stock prices
    """
    # Load model and scaler
    try:
        model = load_model(f'models/{ticker}_model.h5')
        scaler = joblib.load(f'models/{ticker}_scaler.pkl')
    except:
        raise FileNotFoundError(f"No trained model found for {ticker}. Please train first.")
    
    lookback = model.input_shape[1]
    
    # Get the last 'lookback' days of data
    close_prices = data[['Close']].values
    last_sequence = close_prices[-lookback:]
    last_sequence_scaled = scaler.transform(last_sequence)
    
    predictions = []
    current_sequence = last_sequence_scaled.copy()
    
    for _ in range(days_to_predict):
        # Reshape for prediction
        x_input = current_sequence.reshape((1, lookback, 1))
        
        # Predict next day
        pred_scaled = model.predict(x_input, verbose=0)
        
        # Store prediction
        predictions.append(pred_scaled[0, 0])
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], pred_scaled, axis=0)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create dates for predictions
    last_date = data.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    
    return prediction_dates, predictions.flatten()


