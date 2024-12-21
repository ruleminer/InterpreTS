import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_variance(data, ddof=1):
    """
    Calculate the variance value of a time series with specified degrees of freedom.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the variance is to be calculated.
    ddof : int, optional
        Delta degrees of freedom. The divisor used in calculations is N - ddof, where N is the number of elements. 
        A ddof of 1 provides the sample variance, and a ddof of 0 provides the population variance. Default is 1.
        
    Returns
    -------
    float
        The variance of the provided time series with specified degrees of freedom.
        
    Raises
    ------
    TypeError
        If the data is not numeric.
    ValueError
        If the data contains NaN values or is not one-dimensional.
    """
    # Validate the time series without requiring a DateTime index but disallowing NaN
    validate_time_series_data(data, require_datetime_index=False, allow_nan=False)
    
    # Check for numeric data
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")
    
    # Check for one-dimensional data
    if isinstance(data, np.ndarray) and data.ndim != 1:
        raise ValueError("Data must be one-dimensional.")
    if isinstance(data, pd.DataFrame) and data.shape[1] != 1:
        raise ValueError("Data must be one-dimensional.")
    
    # Handle the case where the series has only one value
    if len(data) == 1:
        return 0.0

    # Calculate and return the variance with specified ddof, handling empty series by returning NaN
    return np.var(data, ddof=ddof) if len(data) > 0 else np.nan
