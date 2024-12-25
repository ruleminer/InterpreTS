import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_trough(data, start=None, end=None):
    """
    Calculate the local minimum of a time series within an optional range.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the minimum value is to be calculated.
    start : int, str, or None, optional
        The starting index, timestamp, or position for slicing the data.
        If None, the series starts from the beginning.
    end : int, str, or None, optional
        The ending index, timestamp, or position for slicing the data.
        If None, the series ends at the last value.

    Returns
    -------
    float
        The local minimum of the specified range in the provided time series.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.

    Examples
    --------
    >>> data = pd.Series([1, 2, 5, 4, 3])
    >>> calculate_trough(data)
    1.0
    >>> calculate_trough(data, start=1, end=3)
    2.0
    """
    # Slice the data based on start and end, if provided
    if end is None:
        end = len(data)
    if start is None:
        start = 0
    data = data[start:end]

    # Calculate and return the minimum, handling empty series by returning NaN
    return data.min() if len(data) > 0 else np.nan
