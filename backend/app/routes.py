# routes.py
from flask import Blueprint, jsonify, request
import pandas as pd
import os
from app.models import forecast_oil_price, evaluate_model, plot_forecasted_data

api_bp = Blueprint('api', __name__)

# Base directory for datasets
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Load datasets
datasets = {
    "BrentOilPrices": pd.read_csv(os.path.join(DATA_DIR, 'BrentOilPrices.csv')),
    "InflationUnemployment": pd.read_csv(os.path.join(DATA_DIR, 'inflation_unemployment_data.csv')),
    "CalendarEvents": pd.read_csv(os.path.join(DATA_DIR, 'scraped_calendar_events.csv')),
    "ExchangeRatesAlpha": pd.read_csv(os.path.join(DATA_DIR, 'usd_eur_exchange_rates_alpha_vantage.csv')),
    "ExchangeRatesFred": pd.read_csv(os.path.join(DATA_DIR, 'usd_eur_exchange_rate_fred.csv')),
    "WorldGDPGrowth": pd.read_csv(os.path.join(DATA_DIR, 'world_gdp_growth_data.csv'))
}

@api_bp.route('/forecast', methods=['GET'])
def forecast():
    # Get the start and end date from the request parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({"error": "Please provide both start_date and end_date"}), 400

    # Preprocess the data
    df, df_scaled, scaler = preprocess_data(os.path.join(DATA_DIR, 'BrentOilPrices.csv'))

    # Forecast oil prices
    dates, actual_prices, predicted_prices = forecast_oil_price(df, scaler, start_date, end_date)

    # Evaluate the model
    mse, rmse, mae, r2, mape = evaluate_model(actual_prices, predicted_prices)

    # Plot the forecasted data (you may save the plot to a file if needed)
    plot_forecasted_data(dates, actual_prices, predicted_prices)

    # Return the forecast results and evaluation metrics
    return jsonify({
        "forecasted_dates": dates.tolist(),
        "actual_prices": actual_prices.tolist(),
        "predicted_prices": predicted_prices.tolist(),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape
    })
