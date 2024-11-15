import os
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path, date_column):
    """
    Load a dataset from a CSV file and standardize the date column.
    
    Args:
        file_path (str): Path to the CSV file.
        date_column (str): Name of the column containing date information.
    
    Returns:
        pd.DataFrame: A DataFrame with the date column standardized.
    """
    try:
        df = pd.read_csv(file_path)
        # Rename the date column to 'Date' for consistency
        df.rename(columns={date_column: 'Date'}, inplace=True)
        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def merge_datasets(*dfs):
    """
    Merge multiple datasets on the 'Date' column using an outer join.
    
    Args:
        *dfs: Multiple DataFrames to merge.
    
    Returns:
        pd.DataFrame: A merged DataFrame sorted by date.
    """
    try:
        merged_data = dfs[0]
        for df in dfs[1:]:
            merged_data = merged_data.merge(df, on='Date', how='outer')
        # Sort by 'Date' to ensure data is in chronological order
        return merged_data.sort_values(by='Date')
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        raise

def plot_economic_factors(merged_data):
    """
    Plot various economic factors from a merged DataFrame.
    
    Args:
        merged_data (pd.DataFrame): The merged DataFrame containing all indicators.
    """
    try:
        plt.figure(figsize=(14, 10))

        # Plot Brent Oil Prices
        plt.subplot(2, 2, 1)
        plt.plot(merged_data['Date'], merged_data['Price'], label='Brent Oil Price', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Brent Oil Prices')
        plt.legend()

        # Plot Inflation and Unemployment rate
        plt.subplot(2, 2, 2)
        plt.plot(merged_data['Date'], merged_data['Inflation (annual %)'], label='Inflation (annual %)', color='green')
        plt.plot(merged_data['Date'], merged_data['Unemployment rate (%)'], label='Unemployment rate (%)', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Percentage')
        plt.title('Inflation and Unemployment Rate')
        plt.legend()

        # Plot USD/EUR Exchange Rate (FRED)
        plt.subplot(2, 2, 3)
        plt.plot(merged_data['Date'], merged_data['DEXUSEU'], label='USD/EUR Exchange Rate (FRED)', color='purple')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.title('USD/EUR Exchange Rate (FRED)')
        plt.legend()

        # Plot Exchange Rates (Alpha Vantage)
        plt.subplot(2, 2, 4)
        plt.plot(merged_data['Date'], merged_data['Open'], label='Open', color='red')
        plt.plot(merged_data['Date'], merged_data['High'], label='High', color='pink')
        plt.plot(merged_data['Date'], merged_data['Low'], label='Low', color='brown')
        plt.plot(merged_data['Date'], merged_data['Close'], label='Close', color='gray')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.title('USD/EUR Exchange Rates (Alpha Vantage)')
        plt.legend()

        plt.tight_layout()
        plt.show()
        logger.info("Plots created successfully.")

    except Exception as e:
        logger.error(f"Error plotting economic factors: {e}")
        raise
