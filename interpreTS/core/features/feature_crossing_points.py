import numpy as np
import pandas as pd
from interpreTS.utils.data_validation import validate_time_series_data

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

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 3, 2, 4, 1, 5, 2, 6])
    >>> calculate_crossing_points(data)
    {'crossing_count': 7, 'crossing_points': [0, 1, 2, 3, 4, 5, 6]}

    >>> data = pd.Series([2, 2, 2, 2])
    >>> calculate_crossing_points(data)
    {'crossing_count': 0, 'crossing_points': []}
    """
    # Return immediately if data is empty
    if isinstance(data, (pd.Series, pd.DataFrame)) and data.empty:
        return {'crossing_count': 0, 'crossing_points': []}

    # Convert to numpy array if data is a pandas Series
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    mean_value = np.mean(data)

    # Return 0 crossings if all values are strictly above or below the mean
    epsilon = 1e-10  # tolerancja precyzji
    if np.all(data < mean_value + epsilon) or np.all(data > mean_value - epsilon):
        return {'crossing_count': 0, 'crossing_points': []}


    # Calculate crossings, excluding points equal to mean
    above_mean = data > mean_value
    below_mean = data < mean_value

    # Combine conditions to ignore points equal to mean
    effective_above_mean = np.where(above_mean, 1, np.where(below_mean, -1, 0))

    # Detect crossings by analyzing changes between -1 and 1
    crossings = np.where(np.diff(effective_above_mean) != 0)[0]

    return {
        'crossing_count': len(crossings),
        'crossing_points': list(crossings)
    }
