import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot

# Function to get data shape and descriptive statistics
def get_data_info(df):
    print(df.head())
    print("Data Shape:", df.shape)
    print("Descriptive Statistics:\n", df.describe())

# Function to check for missing values
def check_missing_values(df):
    missing_values = df.isnull().sum()
    print(f"Missing Values:\n{missing_values}")

# Function to prepare data and visualize outliers
def prepare_and_visualize_data(df):
    # Specify the date format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
    
    # Ensure 'Price' is a float
    df['Price'] = df['Price'].astype(float)
    
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    
    # Visualize the data to check for outliers
    plt.figure(figsize=(12, 6))
    sns.boxplot(y=df['Price'])
    plt.title('Boxplot of Oil Prices')
    plt.show()

def feature_engineering(df):
    # Example of creating new features from the Date index
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    print("Feature Engineering", df.head()) 

def plot_price_trend(df):
     # Plotting the time series of Brent Oil Prices
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Brent Oil Price', color='blue')
    plt.title('Brent Oil Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot the time series of Brent Oil Prices with weekly, monthly, and yearly trends
def plot_trends(df):
    # Resample the data to get weekly, monthly, and yearly trends
    weekly_trend = df['Price'].resample('W').mean()  # Weekly average
    monthly_trend = df['Price'].resample('M').mean() # Monthly average
    yearly_trend = df['Price'].resample('Y').mean()  # Yearly average

    # Plotting the original Brent Oil Prices and the trends
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Daily Brent Oil Price', color='blue', linewidth=0.5)
    plt.plot(weekly_trend.index, weekly_trend, label='Weekly Trend', color='orange', linewidth=1.5)
    plt.plot(monthly_trend.index, monthly_trend, label='Monthly Trend', color='green', linewidth=2)
    plt.plot(yearly_trend.index, yearly_trend, label='Yearly Trend', color='red', linewidth=2.5)

    # Add labels and title
    plt.title('Brent Oil Prices with Weekly, Monthly, and Yearly Trends')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.legend()  # Show the legend
    plt.grid()
    plt.show()

# cheking Stationarity
def rolling_mean_moving_average(df):
    # Calculate rolling means with different windows (daily, weekly, monthly, yearly)
    daily_rolling_mean = df['Price'].rolling(window=1).mean()     # Daily rolling mean
    weekly_rolling_mean = df['Price'].rolling(window=7).mean()    # Weekly rolling mean
    monthly_rolling_mean = df['Price'].rolling(window=30).mean()  # Approximate monthly rolling mean
    yearly_rolling_mean = df['Price'].rolling(window=365).mean()  # Approximate yearly rolling mean

    # Plot rolling means for different windows
    plt.figure(figsize=(14, 7))
    plt.legend(title='Simple Moving Averages')
    plt.plot(df['Price'], label='Original Price', color='blue', alpha=0.5)
    plt.plot(daily_rolling_mean, label='Daily Rolling Mean', color='cyan', linewidth=0.8)
    plt.plot(weekly_rolling_mean, label='Weekly Rolling Mean', color='orange', linewidth=1.5)
    plt.plot(monthly_rolling_mean, label='Monthly Rolling Mean', color='green', linewidth=2)
    plt.plot(yearly_rolling_mean, label='Yearly Rolling Mean', color='red', linewidth=2.5)

    plt.title('Brent Oil Price with Daily, Weekly, Monthly, and Yearly Rolling Means')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.legend(title='Simple Moving Averages')
    # plt.legend()
    plt.grid()
    plt.show()

def rolling_std_moving_average(df):
    # Calculate rolling standard deviations with different windows (daily, weekly, monthly, yearly)
    daily_rolling_std = df['Price'].rolling(window=1).std()       # Daily rolling std
    weekly_rolling_std = df['Price'].rolling(window=7).std()      # Weekly rolling std
    monthly_rolling_std = df['Price'].rolling(window=30).std()    # Approximate monthly rolling std
    yearly_rolling_std = df['Price'].rolling(window=365).std()    # Approximate yearly rolling std

    # Plot rolling standard deviations for different windows
    plt.figure(figsize=(14, 7))
    plt.plot(daily_rolling_std, label='Daily Rolling Std', color='cyan', linewidth=0.8)
    plt.plot(weekly_rolling_std, label='Weekly Rolling Std', color='orange', linewidth=1.5)
    plt.plot(monthly_rolling_std, label='Monthly Rolling Std', color='green', linewidth=2)
    plt.plot(yearly_rolling_std, label='Yearly Rolling Std', color='red', linewidth=2.5)

    plt.title('Rolling Standard Deviation of Brent Oil Price (Daily, Weekly, Monthly, and Yearly)')
    plt.xlabel('Date')
    plt.ylabel('Price Standard Deviation (USD per Barrel)')
    plt.legend(title='Standard deviation Moving Averages')
    plt.grid()
    plt.show()

# Histogram distrbution
def distrbution(df):
    # Histogram of Prices
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Price'], bins=30, kde=True)
    plt.title('Distribution of Brent Oil Prices')
    plt.xlabel('Price (USD per Barrel)')
    plt.ylabel('Frequency')
    plt.show()

# Function for Seasonal Decomposition of Time Series (daily, half-yearly, and yearly)
def perform_seasonal_decomposition(df):
    # Check for NaN or infinite values in 'Price' column and handle them
    if df['Price'].isnull().sum() > 0 or (df['Price'] == float('inf')).sum() > 0:
        print("Warning: Data contains NaN or infinite values. Cleaning data.")
        # Drop rows with NaN values in 'Price' or fill them (you can choose the strategy)
        df['Price'] = df['Price'].replace([float('inf'), -float('inf')], float('nan')).dropna()

    # Ensure the data is regularly spaced (e.g., daily frequency)
    df = df.resample('D').mean()  # Adjust this if your data is not daily

    # Decompose the daily time series
    try:
        result = seasonal_decompose(df['Price'], model='multiplicative', period=12)
        plt.figure()
        result.plot()
        plt.suptitle("Daily Seasonal Decomposition", fontsize=16)
        plt.show()

        # Half-Yearly decomposition (assuming monthly data, 6 months in half a year)
        half_yearly_result = seasonal_decompose(df['Price'], model='multiplicative', period=730)
        plt.figure()
        half_yearly_result.plot()
        plt.suptitle("Half-Yearly Seasonal Decomposition", fontsize=16)
        plt.show()

        # Yearly decomposition (assuming daily data and approx. 365 days in a year)
        yearly_result = seasonal_decompose(df['Price'], model='multiplicative', period=365)
        plt.figure()
        yearly_result.plot()
        plt.suptitle("Yearly Seasonal Decomposition", fontsize=16)
        plt.show()

    except ValueError as e:
        print(f"Error during seasonal decomposition: {e}")


def auto_partial_authocorrelation(df):
    # Plot Autocorrelation Function (ACF)
    plt.figure(figsize=(12, 6))
    plot_acf(df['Price'], lags=30)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

    # Plot Partial Autocorrelation Function (PACF)
    plt.figure(figsize=(12, 6))
    plot_pacf(df['Price'], lags=30)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

def simple_moving_average(df):
    # Calculate the Simple Moving Average (SMA) with different window sizes

    df['SMA_3'] = df['Price'].rolling(window=3).mean()  # 3-month moving average
    df['SMA_6'] = df['Price'].rolling(window=6).mean()  # 6-month moving average
    df['SMA_12'] = df['Price'].rolling(window=12).mean()  # 12-month moving average

    # Plot the original data and the moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Original Data', color='blue')
    plt.plot(df.index, df['SMA_3'], label='3-Month Moving Average', color='orange')
    plt.plot(df.index, df['SMA_6'], label='6-Month Moving Average', color='green')
    plt.plot(df.index, df['SMA_12'], label='12-Month Moving Average', color='red')

    # Add titles and labels
    plt.title('Moving Averages (1987-2024)')
    plt.xlabel('Year')
    plt.ylabel('Import Value')
    plt.legend()
    plt.grid(False)
    plt.show()

def lag_plotting(df):
    # Plot lag plot
    plt.figure(figsize=(8, 8))
    lag_plot(df['Price'], lag=1)
    plt.title('Lag Plot (Lag=1)')
    plt.xlabel('Price(t)')
    plt.ylabel('Price(t-1)')
    plt.grid()
    plt.show()    

