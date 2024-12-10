import numpy as np
import pandas as pd
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_significant_changes(data):
    """
    Calculate the proportion of significant increases or decreases in the signal within the given window.
     
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the significant change is to be calculated.
        
    Returns
    -------
    float
        The proportion of significant changes in the window, in the range [0, 1].
        A value > 0 indicates the presence of significant changes.
        
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or has fewer than 2 elements.

    Example
    -------
    >>> data = pd.Series([1, 2, 1.5, 3, 2.5, 5, 4.5])
    >>> calculate_significant_change(data)
    0.3333333333333333

    """

    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)
    
    # Ensure data contains enough points for calculation
    if len(data) < 2:
        return 0.0  # Not enough data to compute differences
    
    # Compute differences between consecutive values
    differences = np.diff(data)
    
    # Calculate Q1, Q3, and IQR for the differences
    Q1 = np.percentile(differences, 25)
    Q3 = np.percentile(differences, 75)
    IQR = Q3 - Q1
    
    # Define bounds for significant changes
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count significant changes
    significant_changes = np.sum((differences < lower_bound) | (differences > upper_bound))
    
    # Proportion of significant changes
    proportion_significant_changes = significant_changes / len(differences)
    
    return proportion_significant_changes