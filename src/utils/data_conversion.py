import pandas as pd
import numpy as np
from ..core.time_series_data import TimeSeriesData

def convert_to_time_series(data):
    """
    Convert input data to a TimeSeriesData object.
    
    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or np.ndarray
        The data to be converted into a TimeSeriesData object.
        
    Returns
    -------
    TimeSeriesData
        An instance of TimeSeriesData wrapping the input data.
        
    Raises
    ------
    TypeError
        If the input data is not of type pd.DataFrame, pd.Series, or np.ndarray.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> ts_data = convert_to_time_series(data)
    """
    
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return TimeSeriesData(data)
    elif isinstance(data, np.ndarray):
        return TimeSeriesData(pd.Series(data))
    else:
        raise TypeError("Data must be of type pandas DataFrame, Series, or numpy array.")
