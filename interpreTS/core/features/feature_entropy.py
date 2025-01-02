import numpy as np
from scipy.stats import gaussian_kde

def calculate_entropy(data):
    """
    Calculate the normalized Shannon entropy of a dataset.

    This function estimates the probability density of the data using Kernel Density Estimation (KDE)
    and calculates the Shannon entropy based on the estimated probabilities.

    Parameters
    ----------
    data : array-like
        A 1D array or list of numerical data points.

    Returns
    -------
    float
        The normalized Shannon entropy of the dataset. If the dataset consists of identical values 
        (i.e., no variability), the entropy is 0. If the KDE results in zero probability for any 
        point, NaN is returned to indicate that the entropy could not be calculated properly.

    Notes
    -----
    - If the range (peak-to-peak value) of the data is zero (i.e., all values are identical), the
      entropy is directly returned as 0.
    - The dataset is evaluated at 100 evenly spaced points between the minimum and maximum values
      of the data for KDE estimation.
    - The Shannon entropy is normalized by dividing by `np.log2(len(x))`, where `len(x)` is the
      number of points used for KDE evaluation, to scale the entropy between 0 and 1.
    - If any probability in the KDE is zero, the function returns NaN, indicating a problematic
      or poorly estimated probability distribution.

    Examples
    --------
    >>> data = [1, 1, 1, 1, 2, 2, 2]
    >>> calculate_entropy(data)
    0.9182958340544894  # Example output, depending on the data distribution.
    """
    
    if len(data) == 0: 
        return np.nan
    
    if np.ptp(data) == 0:
        return 0.0
    
    x = np.linspace(min(data), max(data), 100)
    
    probabilities = gaussian_kde(data)(x)
    probabilities /= probabilities.sum()
    if np.any(probabilities == 0):
        return np.nan
    
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    return shannon_entropy / np.log2(len(x))