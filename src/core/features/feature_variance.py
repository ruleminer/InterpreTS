import pandas as pd
import numpy as np
from src.utils.data_validation import validate_time_series_data

def calculate_variance(data):
    """
    Calculate the variance value of a time series.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the variance is to be calculated.
        
    Returns
    -------
    float
        The variance of the provided time series.
        
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
    >>> calculate_variance(data)
    2.0
    """
    # Validate the time series without requiring a DateTime index
    validate_time_series_data(data, require_datetime_index=False)
    
    # Calculate and return the variance, handling empty series by returning NaN
    return np.var(data) if len(data) > 0 else np.nan
