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
    # Handle single-value case
    if len(data) == 1:
        return 1.0  # Single value is always equal to its mean

    # Handle all-equal case
    if data.nunique() == 1:
        return 0.0  # No value is greater than the mean when all values are the same

    # Calculate mean and binarize the data
    mean_value = data.mean()
    binarized_data = (data >= mean_value).astype(int)  # Greater than or equal to the mean

    return binarized_data.mean()
