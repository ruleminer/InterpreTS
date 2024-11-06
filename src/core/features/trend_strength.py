import pandas as pd
import numpy as np
from scipy.stats import linregress
from src.utils.data_validation import validate_time_series_data

def calculate_trend_strength(data):
    """
    Calculate the strength of the trend in a time series using linear regression.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the trend strength is to be calculated.
        
    Returns
    -------
    float
        The R-squared value representing the strength of the trend (0 to 1).
        
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_trend_strength(data)
    1.0
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)
    
    # Handle empty or insufficient data by returning NaN
    if len(data) < 2:
        return np.nan
    
    # Generate an index as a proxy for time
    x = np.arange(len(data))
    
    # Fit a linear regression and calculate the R-squared value
    slope, intercept, r_value, p_value, std_err = linregress(x, data)
    
    # R-squared value as trend strength
    trend_strength = r_value ** 2
    
    return trend_strength