# app.py
from flask import Flask, render_template, request, redirect, url_for
from utils import get_stock_data, predict_future_prices
from model import train_stock_model
import pandas as pd
import os

app = Flask(__name__)
# Add this debug code
print("Current directory:", os.getcwd())
print("Templates exists:", os.path.exists('templates'))
print("Results.html exists:", os.path.exists('templates/results.html'))


# List of popular stock tickers
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'DIS', 'NFLX', 'PYPL', 'ADBE', 'INTC'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        action = request.form['action']
        
        if action == 'predict':
            return redirect(url_for('predict', ticker=ticker))
        elif action == 'train':
            return redirect(url_for('train', ticker=ticker))
    
    return render_template('index.html', tickers=POPULAR_TICKERS)

@app.route('/train/<ticker>')
def train(ticker):
    try:
        # Train the model (this might take a while)
        model, history, scaler = train_stock_model(ticker, epochs=50)
        message = f"Successfully trained model for {ticker}"
    except Exception as e:
        message = f"Error training model for {ticker}: {str(e)}"
    
    return render_template('results.html', 
                         ticker=ticker,
                         message=message,
                         action="Training")

@app.route('/predict/<ticker>')
def predict(ticker):
    try:
        # Get data
        data = get_stock_data(ticker)
        
        # Make predictions
        dates, predictions = predict_future_prices(ticker, data, days_to_predict=30)
        
        # Prepare data for chart
        history = data['Close'].reset_index()
        history.columns = ['Date', 'Price']
        
        future = pd.DataFrame({'Date': dates, 'Price': predictions})
        
        # Convert to list of dicts for JSON
        history_data = history.tail(100).to_dict('records')  # Last 100 days
        future_data = future.to_dict('records')
        
        return render_template('results.html', 
                             ticker=ticker,
                             history_data=history_data,
                             future_data=future_data,
                             action="Prediction")
    
    except Exception as e:
        return render_template('results.html', 
                             ticker=ticker,
                             message=f"Error predicting for {ticker}: {str(e)}",
                             action="Prediction")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    app.run(debug=True)