import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_below_1st_decile(data, training_data):
    """
    Calculate the fraction of values in the window below the 1st decile of the training data.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the fraction is to be calculated.
    training_data : pd.Series or np.ndarray
        The training data to determine the 1st decile.
        
    Returns
    -------
    float
        The fraction of values in the window below the 1st decile, in the range [0, 1].
        
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or is empty.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> calculate_below_1st_decile(data, training_data)
    0.2
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)
    validate_time_series_data(training_data, require_datetime_index=False)
    
    # Ensure data is not empty
    if len(data) == 0 or len(training_data) == 0:
        raise ValueError("Data and training data must not be empty.")
    
    # Ensure no NaN values in the input
    if np.any(pd.isna(data)) or np.any(pd.isna(training_data)):
        raise ValueError("Data and training data must not contain NaN values.")
    
    # Convert to NumPy arrays for consistency
    data = np.asarray(data)
    training_data = np.asarray(training_data)
    
    # Calculate the 1st decile of the training data
    first_decile = np.percentile(training_data, 10)
    
    # Calculate the fraction of values below the 1st decile
    below_decile_count = np.sum(data < first_decile)
    below_decile_fraction = below_decile_count / len(data)
    
    return below_decile_fraction
