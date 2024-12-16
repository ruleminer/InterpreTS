import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_entropy(data, bins=2):
    """
    Calculate the entropy of a time series based on its value distribution.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the entropy is to be calculated.
    bins : int, optional
        The number of bins to use for discretizing the data. Default is 2.

    Returns
    -------
    float
        The normalized entropy, ranging from 0 to 1.

    Raises
    ------
    TypeError
        If the data is not numeric or not a valid time series type.
    ValueError
        If the data contains NaN values, is too short, or bins < 2.
    """
    # Validate data, disallowing NaN values
    validate_time_series_data(data, allow_nan=False)

    # Ensure data is numeric
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")

    if len(data) < 2:
        raise ValueError("Input data is too short to calculate entropy.")
    if bins < 2:
        raise ValueError("Number of bins must be at least 2.")

    # Check for constant values
    if np.ptp(data) == 0:
        return 0.0  # No entropy for constant values

    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(len(unique))
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy
