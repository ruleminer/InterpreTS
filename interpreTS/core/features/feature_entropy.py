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
    
    print(f"Input data: {data}, bins: {bins}")
    if len(data) < 2:
        print("Too few values for entropy calculation.")
        return np.nan
    if np.ptp(data) == 0:
        print("Constant values, entropy is 0.")
        return 0.0

    unique_values = len(np.unique(data))
    if bins > unique_values:
        bins = unique_values
        print(f"Adjusted bins: {bins}")

    counts, _ = np.histogram(data, bins=bins, density=False)
    probabilities = counts / len(data)
    print(f"Probabilities: {probabilities}")

    if np.any(probabilities == 0):
        probabilities = probabilities[probabilities > 0]

    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(len(probabilities))
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    print(f"Entropy: {normalized_entropy}")
    return normalized_entropy