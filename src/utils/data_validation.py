import pandas as pd
import numpy as np

def validate_time_series_data(data):
    """
    Validate if the input data is suitable for time series processing.
    
    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or np.ndarray
        The time series data to be validated.
        
    Returns
    -------
    bool
        True if the data is valid; raises an error otherwise.
        
    Raises
    ------
    TypeError
        If data is not a pd.Series, pd.DataFrame, or np.ndarray.
    ValueError
        If the data contains NaN values or if pd.Series/pd.DataFrame has no DateTime index.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> validate_time_series_data(data)
    True
    """
    
    
    if not isinstance(data, (pd.Series, pd.DataFrame, np.ndarray)):
        raise TypeError("Data must be a pandas Series, DataFrame, or numpy array.")
    
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if data.isnull().any().any():
            raise ValueError("Data contains NaN values.")
        if isinstance(data, pd.Series) and data.index.is_all_dates:
            pass
        elif isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
            pass
        else:
            raise ValueError("DataFrame or Series must have a DateTime index for time-based operations.")
    
    return True
