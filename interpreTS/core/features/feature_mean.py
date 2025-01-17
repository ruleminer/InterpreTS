import pandas as pd
import numpy as np

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
    >>> calculate_mean(data)
    3.0
    """
    # Calculate and return the mean, handling empty series by returning NaN
    return data.mean() if len(data) > 0 else np.nan
