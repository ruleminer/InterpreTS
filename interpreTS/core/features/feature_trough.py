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
        If the data contains only NaN values or is invalid.

    Examples
    --------
    >>> data = pd.Series([1, 2, 5, 4, 3])
    >>> calculate_trough(data)
    1.0
    >>> calculate_trough(data, start=1, end=3)
    2.0
    """

    # Validate the input type
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError("Data must be a pandas Series or numpy ndarray.")

    # Convert ndarray to pandas Series for consistency
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Validate the time series (ignoring datetime requirement)
    validate_time_series_data(data, require_datetime_index=False)

    # Check if data contains only NaN or is empty
    if data.isna().all() or len(data) == 0:
        return np.nan

    # Adjust slicing parameters
    if start is None:
        start = 0
    if end is None or end > len(data):
        end = len(data)

    # Ensure slicing indices are valid
    if not (0 <= start < len(data)) or not (start < end <= len(data)):
        raise ValueError("Invalid range for start and end indices.")

    # Slice the data
    sliced_data = data.iloc[start:end]

    # Return the minimum value, handling empty slices
    return sliced_data.min() if not sliced_data.empty else np.nan