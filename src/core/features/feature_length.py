from sktime.utils.validation.series import check_series

def calculate_feature_length(data):
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
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_length(data)
    5
    """
    
    
    # Check if the data is empty
    if len(data) == 0:
        return 0
    
    # Validate the time series using sktime's validation tools
    data = check_series(data)
    
    return len(data)