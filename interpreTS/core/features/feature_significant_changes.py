import numpy as np
import pandas as pd

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

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 1, 3, 10, 2, 1])
    >>> calculate_significant_changes(data)
    0.0
    """
    # Convert np.ndarray to pd.Series
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Ensure data contains enough points for calculation
    if len(data) < 2:
        return 0.0  # Not enough data to compute differences
    
    # Compute differences between consecutive values
    differences = np.diff(data)

    # Check for all differences being zero or constant
    if np.all(differences == 0) or np.all(differences == differences[0]):
        return 0.0

    # Use absolute values of differences to handle negative data
    abs_differences = np.abs(differences)

    # Calculate Q1, Q3, and IQR for the absolute differences
    Q1 = np.percentile(abs_differences, 25)
    Q3 = np.percentile(abs_differences, 75)
    IQR = Q3 - Q1

    # Avoid issues with very small IQR
    if IQR == 0:
        IQR = np.abs(Q1) * 0.1 if Q1 != 0 else 0.1  # Safe minimum threshold

    # Define bounds for significant changes
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count significant changes
    significant_changes = np.sum((abs_differences < lower_bound) | (abs_differences > upper_bound))

    # Proportion of significant changes
    proportion_significant_changes = significant_changes / len(differences)

    return proportion_significant_changes
