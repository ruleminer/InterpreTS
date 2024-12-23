import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_stability(data, max_lag=None):
    """
    Calculate the stability of a time series based on autocorrelation.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the stability is to be calculated.
    max_lag : int, optional
        The maximum number of lags to consider for autocorrelation.
        If None, it will be set to `min(12, len(data) - 1)`.

    Returns
    -------
    float
        The stability strength, ranging from 0 to 1, where 1 indicates high stability.
    """
    # Convert to pandas Series if it's a numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Ensure the data is numeric and handle empty series
    if data.empty:
        return np.nan
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")

    # Dynamically determine max_lag if not provided
    if max_lag is None:
        max_lag = min(12, len(data) - 1)

    # Handle insufficient data
    if len(data) <= max_lag or max_lag < 1:
        return np.nan

    # Handle zero variance
    if data.var() == 0:
        return 1.0

    try:
        # Calculate the autocorrelation of the data up to the max lag
        autocorr_values = acf(data, nlags=max_lag, fft=True)

        # Exclude the first autocorrelation (lag 0) as it is always 1
        autocorr_values = autocorr_values[1:]

        # Calculate stability based on variance of autocorrelation values at higher lags
        mean_autocorr = np.mean(np.abs(autocorr_values))
        variance_autocorr = np.var(autocorr_values)

        # Combine mean and variance to get a measure of stability
        stability_strength = 1 - (mean_autocorr + variance_autocorr) / 2
        stability_strength = max(0, min(stability_strength, 1))  # Ensure result is within [0, 1]

        return stability_strength

    except Exception as e:
        # Log unexpected exceptions and return NaN
        print(f"Error calculating stability: {e}")
        return np.nan
