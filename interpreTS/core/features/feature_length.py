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
    # Validate the time series without requiring a DateTime index
    validate_time_series_data(data, require_datetime_index=False)
    
    # Check if the data is one-dimensional
    if isinstance(data, np.ndarray) and data.ndim != 1:
        raise ValueError("Data must be one-dimensional.")
    if isinstance(data, pd.Series) and len(data.shape) != 1:
        raise ValueError("Data must be one-dimensional.")

    # Return the length of the data
    return len(data)
