import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_quantile(data, quantile=0.5):
    """
    Calculates the value for a specified quantile level in the distribution of a time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the quantile is to be calculated.
    quantile : float, optional
        The quantile level (from 0 to 1). Default is 0.5 (median).
        
    Returns
    -------
    float
        The value for the specified quantile level in the data distribution.
        
    Raises
    ------
    TypeError
        If the data is not numeric.
    ValueError
        If the data contains NaNs, is not one-dimensional, or the quantile is outside the range [0, 1].

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> calculate_quantile(data, quantile=0.25)
    3.0
    >>> calculate_quantile(data, quantile=0.5)
    5.0
    >>> calculate_quantile(data, quantile=0.75)
    7.0
    >>> calculate_quantile(data, quantile=1)
    9.0
    >>> calculate_quantile(data, quantile=0)
    1.0
    """
    

    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")
    

    if isinstance(data, np.ndarray) and data.ndim != 1:
        raise ValueError("Data must be one-dimensional.")
    if isinstance(data, pd.DataFrame) and data.shape[1] != 1:
        raise ValueError("Data must be one-dimensional.")
    

    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")

    if not (0 <= quantile <= 1):
        raise ValueError("Quantile level must be in the range [0, 1].")
    

    if len(data) > 0:
        return np.nanquantile(data, quantile)
    else:
        return np.nan