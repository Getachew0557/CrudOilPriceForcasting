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

from changing_point_analysis import (cusum_analysis, bayesian_change_point_detection, 
                                     plot_change_point, detect_change_points, 
                                     plot_detected_change_points,
                                     plot_detected_changing_point_event)

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

    cusum_analysis(df)
    bayesian_change_point_detection(df)
    # Plotting the change point at a specific index
    change_point_index = 4753  # Example change point index
    plot_change_point(df, change_point_index)
    
    # Detecting change points using PELT method
    change_points = detect_change_points(df)
    
    # Plotting the detected change points
    plot_detected_change_points(df, change_points)

    # Plotting detected change points along with events
    plot_detected_changing_point_event(df, change_points)

if __name__ == "__main__":
    main()