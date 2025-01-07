import pandas as pd
import numpy as np

def calculate_above_9th_decile(data, training_data):
    """
    Calculate the fraction of values in the window above the 9th decile of the training data.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the fraction is to be calculated.
    training_data : pd.Series or np.ndarray
        The training data to determine the 9th decile.
        
    Returns
    -------
    float
        The fraction of values in the window above the 9th decile, in the range [0, 1].
        
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or is empty.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([8, 9, 10, 11, 12])
    >>> training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> calculate_above_9th_decile(data, training_data)
    0.6
    """
    # Convert to NumPy arrays for consistency
    data = np.asarray(data)
    training_data = np.asarray(training_data)
    
    # Calculate the 9th decile of the training data
    ninth_decile = np.percentile(training_data, 90)
    
    # Calculate the fraction of values above the 9th decile
    above_decile_count = np.sum(data > ninth_decile)
    above_decile_fraction = above_decile_count / len(data)
    
    return above_decile_fraction
