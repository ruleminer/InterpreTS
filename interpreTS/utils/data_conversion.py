import pandas as pd
import numpy as np
from ..core.time_series_data import TimeSeriesData

def convert_to_time_series(data):
    """
    Convert input data to a TimeSeriesData object.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or np.ndarray
        The data to be converted into a TimeSeriesData object.

    Returns
    -------
    TimeSeriesData
        An instance of TimeSeriesData wrapping the input data.

    Raises
    ------
    TypeError
        If the input data is not of type pd.DataFrame, pd.Series, or np.ndarray.
    ValueError
        If the input data is empty or has invalid dimensions.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> ts_data = convert_to_time_series(data)
    """

    # Handle pandas Series or DataFrame
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return TimeSeriesData(data)

    # Handle numpy array
    elif isinstance(data, np.ndarray):
        if data.size == 0:
            return TimeSeriesData(pd.Series(data))
        if data.ndim > 2:
            raise ValueError("Input numpy array must be 1D or 2D.")
        if data.ndim == 1:
            return TimeSeriesData(pd.Series(data))
        else:
            return TimeSeriesData(pd.DataFrame(data))

    # Raise error for unsupported types
    else:
        raise TypeError("Data must be of type pandas DataFrame, Series, or numpy array.")
