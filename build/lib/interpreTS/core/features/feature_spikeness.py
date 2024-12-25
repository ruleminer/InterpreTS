import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_spikeness(data):
    """
    Calculate the spikeness (skewness) of a time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the spikeness is to be calculated.

    Returns
    -------
    float
        The spikeness (skewness) of the provided time series.

    Raises
    ------
    TypeError
        If the data is not a valid time series type or contains non-numeric values.
    ValueError
        If the data is empty.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_spikeness(data)
    0.0
    """
    # Ensure data is a pandas Series for compatibility with .skew()
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Drop NaN values to avoid issues with skewness calculation
    original_length = len(data)
    data = data.dropna()

    # Handle empty series after dropping NaNs
    if len(data) == 0:
        return np.nan

    # Check if data is numeric
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")

    # Calculate and return spikeness (skewness)
    spikeness = data.skew()

    # Log adjustment if NaN values were dropped
    if len(data) < original_length:
        print(f"Warning: {original_length - len(data)} NaN values were dropped for spikeness calculation.")

    return spikeness
