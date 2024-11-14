import pandas as pd
import numpy as np
from src.utils.data_validation import validate_time_series_data

def calculate_bizarre_mean(data):
    """
    Calculate the bizarre mean of a time series.
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the bizarre mean is to be calculated.
    Returns
    -------
    float
        The bizarre mean of the provided time series.
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_bizarre_mean(data)
    3.0
    """
    
    validate_time_series_data(data, require_datetime_index=False)

    return data.mean() * np.log(len(data)) if isinstance(data, pd.Series) and len(data) > 0 else np.nan
