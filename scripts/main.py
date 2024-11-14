import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from load_data import load_data
from eda_analysis import (get_data_info, check_missing_values, 
                          prepare_and_visualize_data, feature_engineering,
                          plot_price_trend, rolling_std_moving_average,
                           rolling_mean_moving_average,plot_trends,
                            distrbution, perform_seasonal_decomposition,
                             auto_partial_authocorrelation, lag_plotting,
                             simple_moving_average )

def main():
    # Load data
    file_path = '../data/BrentOilPrices.csv'  # Update the path as necessary
    df = load_data(file_path)

    # Perform EDA
    get_data_info(df)
    check_missing_values(df)
    prepare_and_visualize_data(df)
    feature_engineering(df)
    plot_price_trend(df)
    plot_trends(df)
    rolling_mean_moving_average(df)
    rolling_std_moving_average(df)
    distrbution(df)
    perform_seasonal_decomposition(df)
    auto_partial_authocorrelation(df)
    simple_moving_average(df)
    lag_plotting(df)

if __name__ == "__main__":
    main()