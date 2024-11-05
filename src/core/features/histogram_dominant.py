import pandas as pd
import numpy as np
from src.utils.data_validation import validate_time_series_data

def calculate_dominant(data, bins=10):
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
    
    # Check if data is empty after validation
    if len(data) == 0:
        return np.nan
    
    # Calculate histogram and find the bin with the maximum frequency
    counts, bin_edges = np.histogram(data, bins=bins)
    max_bin_index = np.argmax(counts)
    
    # Calculate the center of the bin with the highest frequency
    dominant_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    
    return dominant_value