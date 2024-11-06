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

    Returns
    -------
    float
        The dominant value (mode) of the histogram.
        
    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 1, 2, 3, 3, 3, 4, 5])
    >>> calculate_dominant(data)
    3.0
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)
    
    if len(data) == 0:
        return np.nan
    
    # Get unique values and use them as bin edges
    unique_values = np.unique(data)
    bins = np.append(unique_values, unique_values[-1] + 1)  # Add an extra edge to cover the last bin

    # Calculate histogram with specific bin edges
    counts, bin_edges = np.histogram(data, bins=bins)
    max_bin_index = np.argmax(counts)

    # Calculate the center or lower bound of the bin with the highest frequency
    if return_bin_center:
        dominant_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    else:
        dominant_value = bin_edges[max_bin_index]
    
    return dominant_value