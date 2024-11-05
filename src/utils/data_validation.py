import pandas as pd
import numpy as np

def validate_time_series_data(data, require_datetime_index=False):
    """
    Validate if the input data is suitable for time series processing.
    
    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or np.ndarray
        The time series data to be validated.
    require_datetime_index : bool, optional
        If True, validation will ensure the data has a DateTime index (for time-based operations).
        
    Returns
    -------
    bool
        True if the data is valid; raises an error otherwise.
        
    Raises
    ------
    TypeError
        If data is not a pd.Series, pd.DataFrame, or np.ndarray.
    ValueError
        If the data contains NaN values or lacks a DateTime index when required.
    """
    
    if not isinstance(data, (pd.Series, pd.DataFrame, np.ndarray)):
        raise TypeError("Data must be a pandas Series, DataFrame, or numpy array.")
    
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if data.isnull().any().any():
            raise ValueError("Data contains NaN values.")
        if require_datetime_index:
            if len(data) == 0:  # Allow empty series if datetime index is not required
                return True
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Series must have a DateTime index for time-based operations.")
    
    return True
