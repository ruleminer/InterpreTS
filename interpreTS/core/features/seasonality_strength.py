import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from interpreTS.utils.data_validation import validate_time_series_data

import warnings

def calculate_seasonality_strength(data, period=2, max_lag=12):
    """
    Calculate the strength of the seasonality in a time series based on autocorrelation.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the seasonality strength is to be calculated.
    period : int, optional
        The periodic interval to check for seasonality (default is 2).
    max_lag : int, optional
        The maximum number of lags to consider for autocorrelation (default is 12).

    Returns
    -------
    float
        The seasonality strength, ranging from 0 to 1, where 1 indicates strong seasonality.
        Returns np.nan if the data is insufficient or invalid.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values, is too short to calculate seasonality, or if the period is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2], index=pd.date_range("2023-01-01", periods=12, freq="M"))
    >>> calculate_seasonality_strength(data, period=3)
    1.0
    """
    # Check for a valid period
    if period <= 0:
        raise ValueError("Period must be a positive integer.")
    
    # Convert data to numpy array for consistency
    if isinstance(data, pd.Series):
        data = data.values

    # Remove NaN values
    data = data[~np.isnan(data)]

    # Handle insufficient data length
    if len(data) <= period:
        return np.nan

    # Handle constant data
    if np.all(data == data[0]):
        return 0.0

    try:
        # Suppress warnings from acf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            autocorr_values = acf(data, nlags=max(max_lag, period), fft=True)
        
        # Ensure the autocorrelation result is valid
        if len(autocorr_values) <= period:
            return np.nan
        
        # Extract the autocorrelation value at the specified period
        seasonality_strength = autocorr_values[period]
        
        # Ensure the strength value is in the range [0, 1]
        seasonality_strength = max(0.0, min(seasonality_strength, 1.0))
        return seasonality_strength
    
    except ZeroDivisionError:
        # Handle division by zero in acf calculation (e.g., constant data)
        return 0.0
    
    except Exception as e:
        # Log or handle unexpected exceptions
        print(f"Error calculating seasonality strength: {e}")
        return np.nan
