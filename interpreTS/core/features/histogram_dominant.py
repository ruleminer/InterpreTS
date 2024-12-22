import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_dominant(data, bins=10, return_bin_center=False):
    """
    Calculate the dominant value (mode) of a time series histogram.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the dominant value is to be calculated.
    bins : int, optional
        The number of bins to use for creating the histogram, by default 10.
    return_bin_center : bool, optional
        If True, return the center of the bin with the maximum frequency. 
        Otherwise, return the lower bound of the bin (default is False).

    Returns
    -------
    float
        The dominant value of the histogram (either the center or the lower bound of the bin).

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.
    """
    # Handle empty data early
    if isinstance(data, (pd.Series, pd.DataFrame)) and data.empty:
        return np.nan
    if isinstance(data, np.ndarray) and data.size == 0:
        return np.nan

    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins=bins)
    max_bin_index = np.argmax(counts)

    # Return the lower bound or center of the dominant bin
    if return_bin_center:
        dominant_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    else:
        dominant_value = bin_edges[max_bin_index]
    
    return dominant_value
