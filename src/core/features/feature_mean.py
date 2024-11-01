import pandas as pd
import numpy as np
from src.utils.data_validation import validate_time_series_data

def calculate_mean(data):
    """
    Calculate the mean value of a time series.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the mean value is to be calculated.
        
    Returns
    -------
    float
        The mean value of the provided time series.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_mean(data)
    3.0
    """
    # Validate the time series using sktime's validation tools
    data = validate_time_series_data(data)  
      
    return data.mean()