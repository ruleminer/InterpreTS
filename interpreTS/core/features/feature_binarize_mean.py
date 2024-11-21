import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_binarize_mean(data):
    """
    Calculate the binarize mean of a time series.
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the binarize mean is to be calculated.
    Returns
    -------
    float
        The binarize mean of the provided time series.
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
    >>> calculate_binarize_mean(data)
    0.6
    """
    validate_time_series_data(data, require_datetime_index=False)

    mean_value = data.mean()
    binarized_data = (data > mean_value).astype(int)

    return binarized_data.mean() if isinstance(data, pd.Series) and len(data) > 0 else np.nan
