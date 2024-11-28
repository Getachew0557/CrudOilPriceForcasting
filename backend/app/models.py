# models.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(file_path):
    """
    Preprocess the Brent oil price data: load, convert dates, and scale the data.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Price']])
    return df, df_scaled, scaler

def create_dataset(data, time_step=1):
    """
    Prepare dataset for LSTM input.
    """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_lstm_model(X_train):
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using various metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mae, r2, mape

def forecast_oil_price(df, scaler, start_date, end_date, time_step=60):
    """
    Forecast oil prices for a given date range using LSTM model.
    """
    # Preprocess and split data
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df_scaled = scaler.transform(df_filtered[['Price']])
    
    # Create dataset
    X, y = create_dataset(df_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build and train LSTM model
    model = build_lstm_model(X)
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred)
    
    return df_filtered['Date'], df_filtered['Price'], y_pred

def plot_forecasted_data(dates, actual_prices, predicted_prices):
    """
    Plot the actual vs predicted prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Prices', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Brent Oil Price')
    plt.title('Forecasted Brent Oil Prices')
    plt.legend(loc='upper left')
    plt.show()
