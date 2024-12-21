import numpy as np
import pandas as pd
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_std_1st_der(data):
    """
    Calculate the standard deviation of the first derivative of a time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the standard deviation of the first derivative is to be calculated.

    Returns
    -------
    float
        The standard deviation of the first derivative of the provided time series.
        Returns np.nan if the input data is empty.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.

    Examples
    --------
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_std_1st_der(data)
    0.0
    """
    try:
        # Validate the time series without requiring a DateTime index
        validate_time_series_data(data, require_datetime_index=False, allow_nan=False)
    except ValueError as e:
        # Return np.nan for empty data
        if "Input data is empty." in str(e):
            return np.nan
        raise

    # If there is only one value, return 0.0
    if len(data) == 1:
        return 0.0

    # Calculate and return the standard deviation of the first derivative
    return np.std(np.gradient(data))
