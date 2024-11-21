import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_distance_to_last_trend_change(data, window_size=5):
    """
    Calculate the distance (in terms of indices) to the last trend change point in a time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the distance to the last trend change is to be calculated.
    window_size : int, optional
        The size of the rolling window to calculate mean (default is 5).

    Returns
    -------
    int
        The distance (in terms of indices) to the last trend change point.
        If no change is detected, returns None.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or is too short to calculate mean.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
    >>> calculate_distance_to_last_trend_change(data, window_size=3)
    4
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)

    # Handle empty or insufficient data
    if len(data) < window_size + 1:
        raise ValueError("The time series is too short for the specified rolling window size.")
    
    # Convert data to a pandas Series if it's an ndarray
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Check for monotonic data
    if data.is_monotonic_increasing or data.is_monotonic_decreasing:
        return None  # No trend change can occur in monotonic data
    
    # Calculate rolling mean
    rolling_mean = data.rolling(window=window_size).mean()

    # Calculate the change in mean as the first difference of rolling means
    change_in_mean = rolling_mean.diff()

    # Detect trend change points: where the direction of the change in mean flips
    trend_changes = (change_in_mean > 0) != (change_in_mean.shift(1) > 0)

    # Find the last trend change point
    last_change_point = trend_changes.idxmax() if trend_changes.any() else None
    
    if last_change_point is None:
        return None  # No trend change detected

    # Calculate the distance to the last trend change point
    distance_to_last_change = len(data) - last_change_point

    return distance_to_last_change