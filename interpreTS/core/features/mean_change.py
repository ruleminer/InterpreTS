import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_mean_change(data, window_size=5):
    """
    Calculate the change in mean over time in a time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the change in mean is to be calculated.
    window_size : int, optional
        The size of the rolling window to calculate mean (default is 5).

    Returns
    -------
    pd.Series
        A series containing the change in mean over time, with the same index as the input.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values, is too short for the rolling window, 
        or if the window size is not a positive integer.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> calculate_mean_change(data, window_size=3)
    0     NaN
    1     NaN
    2     NaN
    3     1.00
    4     1.00
    5     1.00
    6     1.00
    7     1.00
    8     1.00
    9     1.00
    dtype: float64
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False, allow_nan=False)

    # Validate window_size
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

    # Handle empty or insufficient data
    if len(data) < window_size + 1:
        raise ValueError("The time series is too short for the specified rolling window size.")

    # Convert data to a pandas Series if it's an ndarray
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Calculate rolling mean
    rolling_mean = data.rolling(window=window_size).mean()

    # Calculate the change in mean as the first difference of rolling means
    change_in_mean = rolling_mean.diff()

    return change_in_mean
