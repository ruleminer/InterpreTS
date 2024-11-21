import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_entropy(data, bins=None):
    """
    Calculate the entropy of a time series based on its value distribution.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the entropy is to be calculated.
    bins : int, optional
        The number of bins to use for discretizing the data. 
        Default is 10.

    Returns
    -------
    float
        The normalized entropy, ranging from 0 to 1, where 0 indicates 
        no entropy (completely predictable) and 1 indicates maximum entropy 
        (completely random).

    Raises
    ------
    TypeError
        If the data is not a valid time series type.
    ValueError
        If the data contains NaN values or is too short to calculate entropy.
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)
    
    # Convert data to numpy array if it's a pandas Series
    if isinstance(data, pd.Series):
        data = data.values
        
    # Handle empty or insufficient data
    if len(data) == 0:
        raise ValueError("The input data is empty.")
    if len(data) < 2:
        raise ValueError("The input data is too short to calculate entropy.")
    
    if bins is None:
        bins = len(data)
    
    # Check for constant values
    if np.all(data == data[0]):
        return 0.0  # No entropy for constant values
    
    # Discretize the data into bins and get the histogram
    hist, _ = np.histogram(data, bins=bins, density=True)
    
    # Normalize histogram to get a probability distribution
    probabilities = hist / np.sum(hist)
    
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate Shannon entropy manually
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Normalize the entropy to range [0, 1]
    max_entropy = np.log2(len(probabilities))
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Ensure the result is within [0, 1]
    normalized_entropy = max(0.0, min(normalized_entropy, 1.0))
    
    return normalized_entropy