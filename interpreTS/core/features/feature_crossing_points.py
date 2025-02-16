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
        If the input data is empty or contains NaN values.
    """
     # Check if data is empty or contains NaN
    if isinstance(data, (pd.Series, pd.DataFrame)) and data.empty:
        return {'crossing_count': 0, 'crossing_points': []}
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    if len(data) == 0 or np.any(np.isnan(data)):
        raise ValueError("Input data should not be empty or contain NaN values.")

    # Calculate the mean value
    mean_value = np.mean(data)

    # If all values are above or all are below the mean, return no crossings
    if np.all(data >= mean_value) or np.all(data <= mean_value):
        return {'crossing_count': 0, 'crossing_points': []}

    crossing_points = []

    # We now check for true crossings
    for i in range(1, len(data)):
        if (data[i-1] < mean_value and data[i] >= mean_value) or (data[i-1] > mean_value and data[i] <= mean_value):
            crossing_points.append(i)

    return {'crossing_count': len(crossing_points), 'crossing_points': crossing_points}