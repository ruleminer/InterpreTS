import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_absolute_energy(data, start=None, end=None):
    """
    Calculate the absolute energy of a time series within an optional range.

    Absolute energy is defined as the sum of squared values in the time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the absolute energy is to be calculated.
    start : int, str, or None, optional
        The starting index, timestamp, or position for slicing the data.
        If None, the series starts from the beginning.
    end : int, str, or None, optional
        The ending index, timestamp, or position for slicing the data.
        If None, the series ends at the last value.

    Returns
    -------
    float
        The absolute energy of the specified range in the provided time series.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4])
    >>> calculate_absolute_energy(data)
    30.0
    >>> calculate_absolute_energy(data, start=1, end=3)
    13.0
    """

    # Validate the time series without requiring a DateTime index
    validate_time_series_data(data, require_datetime_index=False)

    # Slice the data based on start and end, if provided
    if end is None:
        end = len(data)
    if start is None:
        start = 0
    data = data[start:end]

    # Calculate and return the absolute energy, handling empty series by returning NaN
    return np.sum(np.square(data)) if len(data) > 0 else np.nan

