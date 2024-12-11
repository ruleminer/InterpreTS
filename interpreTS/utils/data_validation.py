import pandas as pd
import numpy as np

def validate_time_series_data(data, require_datetime_index=False, allow_nan=True):
    """
    Validate if the input data is suitable for time series processing.
    
    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or np.ndarray
        The time series data to be validated.
    require_datetime_index : bool, optional
        If True, validation will ensure the data has a DateTime index (for time-based operations).
    allow_nan : bool, optional
        If False, validation will raise an error if NaN values are present (default is True).
        
    Returns
    -------
    bool
        True if the data is valid; raises an error otherwise.
        
    Raises
    ------
    TypeError
        If data is not a pd.Series, pd.DataFrame, or np.ndarray.
    ValueError
        If the data contains NaN values and `allow_nan` is False, or if the index is not a DateTime index when required.
    """
    # Check data type
    if not isinstance(data, (pd.Series, pd.DataFrame, np.ndarray)):
        raise TypeError("Data must be a pandas Series, DataFrame, or numpy array.")
    
    # Validate pandas Series or DataFrame
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # Check for NaN values
        if not allow_nan and data.isnull().any().any():
            raise ValueError("Data contains NaN values.")
        
        # Check index type if datetime index is required
        if require_datetime_index and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DateTime index for time-based operations.")
    
    # Validate numpy array
    elif isinstance(data, np.ndarray):
        if data.ndim > 1 and not allow_nan:
            if np.isnan(data).any():
                raise ValueError("Data contains NaN values.")
    
    return True
