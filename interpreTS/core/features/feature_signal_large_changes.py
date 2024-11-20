import numpy as np
import pandas as pd

def significant_change(data, window_size):
    """
    Calculate the significant increase/decrease in the signal within the given window.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the significant change is to be calculated.
    window_size : int
        The size of the window to analyze.
        
    Returns
    -------
    float
        The proportion of significant changes in the window.
        
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or if window_size is invalid.
    """
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError("Data must be a pandas Series or numpy array.")
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values.")
    if window_size <= 0 or window_size > len(data):
        raise ValueError("Invalid window size.")
    
    differences = np.diff(data)
    Q1 = np.percentile(differences, 25)
    Q3 = np.percentile(differences, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    significant_changes = np.sum((differences < lower_bound) | (differences > upper_bound))
    proportion_significant_changes = significant_changes / len(differences)
    
    return proportion_significant_changes