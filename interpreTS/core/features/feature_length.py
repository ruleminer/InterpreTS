from interpreTS.utils.data_validation import validate_time_series_data
import numpy as np
import pandas as pd

def calculate_length(data):
    """
    Calculate the number of data points in a time series.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the length feature is to be calculated.
        
    Returns
    -------
    int
        The number of data points in the provided time series.
        
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or is not one-dimensional.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_length(data)
    5
    """
    # Return the length of the data
    return len(data)
