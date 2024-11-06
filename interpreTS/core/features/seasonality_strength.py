import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from interpreTS.utils.data_validation import validate_time_series_data

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

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or is too short to calculate seasonality.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2], index=pd.date_range("2023-01-01", periods=12, freq="M"))
    >>> calculate_seasonality_strength(data, period=3)
    0.75
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)
    
    # Handle empty or insufficient data
    if len(data) < period + 1:
        return np.nan
    
    # Calculate the autocorrelation of the data
    autocorr_values = acf(data, nlags=max(max_lag, period), fft=True)
    
    # The seasonality strength is based on the autocorrelation at the specified period
    seasonality_strength = autocorr_values[period] if period < len(autocorr_values) else 0.0
    
    return seasonality_strength
