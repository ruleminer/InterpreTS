import numpy as np
import pandas as pd

def calculate_crossing_points(data):
    """
    Calculate the number of times and the list of indices where the time series crosses its mean.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which mean crossings are to be calculated.
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'crossing_count': The total number of crossings.
        - 'crossing_points': A list of indices where crossings occur.
    
    Raises
    ------
    ValueError
        If the data contains NaN values.
    
    Examples
    --------
    >>> data = pd.Series([1, 2, 3, 2, 1, 3, 1, 0])
    >>> calculate_crossing_points(data)
    {'crossing_count': 3, 'crossing_points': [2, 4, 5]}
    """
    if isinstance(data, pd.Series):
        data = data.values  # Convert to np.ndarray for consistency
    
    # Ensure there are no NaN values
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values.")
    
    mean_value = np.mean(data)
    # Boolean array indicating if the data points are above or below the mean
    above_mean = data > mean_value
    # Detect crossings by finding where the difference between consecutive points changes
    crossings = np.where(np.diff(above_mean.astype(int)) != 0)[0] + 1
    
    return {
        'crossing_count': len(crossings),
        'crossing_points': list(crossings)
    }
