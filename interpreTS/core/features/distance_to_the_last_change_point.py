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
    int or None
        The distance (in terms of indices) to the last trend change point.
        If no change is detected, returns None.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or `window_size` is invalid.

    Examples
    --------
    >>> data = pd.Series([1, 2, 3, 2, 1, 2, 3, 2, 1])
    >>> calculate_distance_to_last_trend_change(data, window_size=2)
    1
    """
    # Handle invalid or insufficient data
    if len(data) < window_size + 1:
        raise ValueError("The time series is too short for the specified rolling window size.")
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

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
    trend_changes = ((change_in_mean > 0) != (change_in_mean.shift(1) > 0)) & ~change_in_mean.isna()

    # Get indices of trend change points
    trend_change_indices = trend_changes[trend_changes].index

    if trend_change_indices.empty:
        return None  # No trend change detected

    # Find the last trend change point index
    last_change_index = trend_change_indices[-1]

    # Calculate the distance from the last trend change point to the end of the series
    distance_to_last_change = len(data) - last_change_index - 1

    return distance_to_last_change
