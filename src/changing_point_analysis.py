import pandas as pd
import numpy as np
import seaborn as sns
import ruptures as rpt
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

def cusum_analysis(df):
    """
    Perform CUSUM (Cumulative Sum) analysis on the Brent Oil Prices time series.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the 'Price' column for the analysis.
    """
    # CUSUM method
    print("===============", df.head())
    mean_price = df['Price'].mean()
    cusum = np.cumsum(df['Price'] - mean_price)

    # Plotting the CUSUM values
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, cusum, label='CUSUM')
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
    plt.xlabel('Date')
    plt.ylabel('CUSUM Value')
    plt.title('CUSUM Analysis')
    plt.legend()
    plt.show()


def bayesian_change_point_detection(df):
    """
    Perform Bayesian Change Point Detection using PyMC3 on the Brent Oil Prices time series.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the 'Price' column for the analysis.
    """
    mean_price = df['Price'].mean()

    with pm.Model() as model:
        # Priors
        mean_prior = pm.Normal('mean_prior', mu=mean_price, sigma=10)
        change_point = pm.DiscreteUniform('change_point', lower=0, upper=len(df)-1)  # Change point prior

        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mean_prior, sigma=10, observed=df['Price'])  # Price data likelihood

        # Inference
        trace = pm.sample(1000, tune=1000, cores=2)  # Sampling

    # Plot results of the posterior distribution for the change point
    pm.plot_posterior(trace)
    plt.title('Posterior Distribution for Change Point')
    plt.show()

    # Return the trace for further analysis
    return trace    

def plot_change_point(df, change_point_index):
    """
    Plot the Brent Oil Prices with a specified change point.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the 'Price' column.
    change_point_index (int): Index of the change point in the data.
    """
    # Plot the data with the change point
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Brent Oil Price')
    plt.axvline(x=df.index[change_point_index], color='red', linestyle='--', label='Change Point')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Brent Oil Prices with Change Point')
    plt.legend()
    plt.show()

def detect_change_points(df, model="rbf", pen=20):
    """
    Detect change points in the Brent Oil Price data using the PELT method.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the 'Price' column.
    model (str): The model to be used for segmentation. Default is "rbf" (Radial Basis Function).
    pen (int): Penalty for change point detection.
    
    Returns:
    list: Indices of detected change points.
    """
    price_array = df['Price'].values
    algo = rpt.Pelt(model=model).fit(price_array)
    change_points = algo.predict(pen=pen)
    
    return change_points

def plot_detected_change_points(df, change_points):
    """
    Plot the Brent Oil Prices with detected change points.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the 'Price' column.
    change_points (list): List of indices where change points are detected.
    """
    # Check if change points are valid indices
    valid_change_points = [cp for cp in change_points if cp < len(df) and not pd.isna(df.index[cp])]
    
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Brent Oil Price')

    # Plot each valid change point
    for cp in valid_change_points[:-1]:  # Exclude the last change point as it's the end of the data
        plt.axvline(x=df.index[cp], color='red', linestyle='--', label='Change Point' if cp == valid_change_points[0] else "")

    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Brent Oil Prices with Detected Change Points')
    plt.legend()
    plt.show()


def plot_detected_changing_point_event(df, change_points):
    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Event data dictionary
    event_data = {
        '1999-10-01': 'OPEC production cuts take effect',
        '2004-07-01': 'Hurricane Ivan impacts U.S. production',
        '2005-08-01': 'Hurricane Katrina causes severe supply disruptions',
        '2007-05-01': "OPEC’s output policies influence prices",
        '2008-04-01': 'Global financial crisis begins affecting demand',
        '2008-10-01': 'Economic downturn leads to reduced demand',
        '2009-04-01': 'Early signs of economic recovery',
        '2011-01-03': 'Political unrest in the Middle East (Arab Spring)',
        '2014-10-01': 'OPEC decides not to cut production',
        '2015-01-02': 'Oil price crash due to oversupply',
        '2015-10-01': 'Global economic concerns affect demand',
        '2017-07-03': 'Increased U.S. inventories impact prices',
        '2020-04-01': 'COVID-19 pandemic leads to historic price drop',
        '2020-07-01': 'Recovery in demand as economies reopen',
        '2021-01-04': 'Vaccine rollout boosts global economic outlook',
        '2022-01-03': 'Geopolitical tensions over Ukraine',
    }
    
    # Remove rows where dates couldn't be converted
    df = df.dropna(subset=['Price'])
    
    # Create the figure and plot
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Brent Oil Price', color='blue')

    # Add vertical lines for change points
    for cp in change_points[:-1]:
        # Check if the index at 'cp' is valid
        if pd.notna(df.index[cp]):
            plt.axvline(x=df.index[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else "")

    # Add vertical lines and annotations for events
    for date, event in event_data.items():
        event_date = pd.to_datetime(date)
        if event_date in df.index:
            price = df.loc[event_date, 'Price']
            plt.axvline(event_date, color='green', linestyle='--', alpha=0.6)  # Event lines in green
            plt.text(event_date, price, event, rotation=90, verticalalignment='center', fontsize=8, color='black', fontweight='bold')

    # Add labels and title
    plt.title('Brent Oil Prices with Detected Change Points and Events')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')

    # Add a legend with 'Brent Oil Price' and 'Event' labels
    plt.legend(['Brent Oil Price', 'Change Point', 'Event'], loc='upper left')

    # Grid and layout
    plt.grid()
    plt.tight_layout()  
    plt.show()