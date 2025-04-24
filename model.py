# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_lstm_model(input_shape):
    """
    Build LSTM model architecture
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Third LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Train the LSTM model
    """
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('models/best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


# model.py
from utils import get_stock_data, preprocess_data, train_test_split
import os

def train_stock_model(ticker, lookback=60, epochs=100, batch_size=32, force_download=False):
    """
    Complete training pipeline for a stock ticker
    """
    # Get data
    data = get_stock_data(ticker, force_download=force_download)
    
    # Preprocess data
    X, y, scaler = preprocess_data(data, lookback=lookback)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Build model
    model = build_lstm_model((X_train.shape[1], 1))
    
    # Train model
    model, history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size
    )
    
    # Save the final model and scaler
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{ticker}_model.h5')
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, f'models/{ticker}_scaler.pkl')
    
    return model, history, scaler

    