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
    """
    if len(data) < 2 or np.ptp(data) == 0:
        return 0.0

    # Binning data
    counts, _ = np.histogram(data, bins=bins, density=False)
    probabilities = counts / counts.sum()

    # Avoid log2(0) by filtering probabilities
    probabilities = probabilities[probabilities > 0]

    shannon_entropy = -np.dot(probabilities, np.log2(probabilities))
    max_entropy = np.log2(len(probabilities))
    return shannon_entropy / max_entropy if max_entropy > 0 else 0.0
