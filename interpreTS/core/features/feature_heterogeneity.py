import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_heterogeneity(data):
    """
    Calculate the heterogeneity (coefficient of variation) of a time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the heterogeneity is to be calculated.

    Returns
    -------
    float
        The heterogeneity (coefficient of variation) of the provided time series.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or if the mean of the series is zero.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_heterogeneity(data)
    0.5270462766947299
    """
    # Validate the time series without requiring a DateTime index
    validate_time_series_data(data, require_datetime_index=False)
    
    # Ensure data is a pandas Series for compatibility
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Calculate mean and standard deviation
    mean = data.mean()
    std_dev = data.std()
    
    # Handle empty series
    if len(data) == 0:
        return np.nan
    
    # Handle a single value in the series
    if len(data) == 1:
        return 0.0  # No variability
    
    return std_dev / abs(mean) if mean != 0 else np.nan
