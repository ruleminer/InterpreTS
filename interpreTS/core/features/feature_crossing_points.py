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
    """
    validate_time_series_data(data)

    # Convert to numpy array if data is a pandas Series
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    # Return immediately if data is empty
    if len(data) == 0:
        return {'crossing_count': 0, 'crossing_points': []}

    mean_value = np.mean(data)

    # Return 0 crossings if all values are above or below the mean
    if np.all(data <= mean_value) or np.all(data >= mean_value):
        return {'crossing_count': 0, 'crossing_points': []}

    # Calculate crossings
    above_mean = data > mean_value
    crossings = np.where(np.diff(above_mean.astype(int)) != 0)[0] 

    return {
        'crossing_count': len(crossings),
        'crossing_points': list(crossings)
    }
