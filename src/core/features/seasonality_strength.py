import numpy as np
import pandas as pd 
from src.utils.data_validation import validate_time_series_data

def calculate_seasonality_strength(data, frequency=None):
    """
    Calculate the strength of seasonality in a time series.

    Parameters
    ----------
    data : pd.Series
        The time series data for which the seasonality strength is to be calculated.
    frequency : int, optional
        Expected frequency of the seasonality (in number of data points).

    Returns
    -------
    float
        Seasonality strength, with higher values indicating stronger seasonal patterns.
    """
    validate_time_series_data(data, require_datetime_index=True)

    if len(data) < 2:
        return 0.0

    # Compute the average for each season
    seasonal_means = data.groupby(data.index.dayofweek).mean()

    # Calculate the total mean
    overall_mean = data.mean()

    # Calculate the seasonality strength
    seasonality_strength = ((seasonal_means - overall_mean) ** 2).sum() / (overall_mean ** 2)

    # Debug output for final strength
    print("Seasonal Means:", seasonal_means.values)
    print("Overall Mean:", overall_mean)
    print("Seasonality Strength:", seasonality_strength)

    return seasonality_strength