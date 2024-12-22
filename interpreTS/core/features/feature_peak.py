import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data


def calculate_peak(data, start=None, end=None):
    """
    Calculate the local maximum of a time series within an optional range.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the maximum value is to be calculated.
    start : int, str, or None, optional
        The starting index, timestamp, or position for slicing the data.
        If None, the series starts from the beginning.
    end : int, str, or None, optional
        The ending index, timestamp, or position for slicing the data.
        If None, the series ends at the last value.

    Returns
    -------
    float
        The local maximum of the specified range in the provided time series.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 5, 4, 7])
    >>> calculate_peak(data)
    7.0
    >>> calculate_peak(data, start=1, end=3)
    5.0
    """

    # Validate the time series without requiring a DateTime index
    validate_time_series_data(data, require_datetime_index=False)

    # Slice the data based on start and end, if provided
    if end is None:
        end = len(data)
    if start is None:
        start = 0
    data = data[start:end]
    
    # Calculate and return the maximum, handling empty series by returning NaN
    return data.max() if len(data) > 0 else np.nan
