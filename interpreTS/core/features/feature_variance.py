import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_variance(data, ddof=0):
    """
    Calculate the variance value of a time series with specified degrees of freedom.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the variance is to be calculated.
    ddof : int, optional
        The degrees of freedom to use when calculating the variance. Default is 0 (population variance).
        
    Returns
    -------
    float
        The variance of the provided time series with specified degrees of freedom.
        
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
    2.5
    >>> calculate_variance(data, ddof=0)  # Population variance
    2.0
    """
    # Validate the time series without requiring a DateTime index
    validate_time_series_data(data, require_datetime_index=False)
    
    # Calculate and return the variance with specified ddof, handling empty series by returning NaN
    return np.var(data, ddof=ddof) if len(data) > 0 else np.nan