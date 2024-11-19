import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_below_1st_decile(data, window_size, training_data):
    """
    Calculate the percentage of values in the window below the 1st decile of the training data.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the percentage is to be calculated.
    window_size : int
        The size of the window to calculate the percentage.
    training_data : pd.Series or np.ndarray
        The training data to determine the 1st decile.
        
    Returns
    -------
    float
        The percentage of values in the window below the 1st decile.
        
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
    >>> training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> calculate_below_1st_decile(data, 5, training_data)
    20.0
    """
    
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)
    validate_time_series_data(training_data, require_datetime_index=False)
    
    # Calculate the 1st decile of the training data
    first_decile = np.percentile(training_data, 10)
    
    # Calculate the percentage of values in the window below the 1st decile
    below_decile_count = sum(1 for value in data[:window_size] if value < first_decile)
    percentage_below_decile = (below_decile_count / window_size) * 100
    
    return percentage_below_decile