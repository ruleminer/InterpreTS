import pandas as pd
import numpy as np
from interpreTS.core.features.feature_peak import calculate_peak
from scipy.signal import find_peaks

def calculate_amplitude_change_rate(data):
    """
    Calculate the average amplitude change rate in a time series,
    defined as the mean change in amplitude between consecutive local peaks and troughs.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the amplitude change rate is to be calculated.

    Returns
    -------
    float
        The average amplitude change rate. Returns NaN if no peaks/troughs are found.

    Raises
    ------
    TypeError
        If the data is not numeric or not a valid type.
    ValueError
        If the data contains NaN values or is not one-dimensional.
    """
    # Validate input type
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError("Data must be a pd.Series or np.ndarray.")
    
    # Convert to numpy array for uniform processing
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    # Ensure the data is numeric
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")
    
    # Ensure the data is one-dimensional
    if data.ndim != 1:
        raise ValueError("Data must be one-dimensional.")
    
    # Check for NaN values
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    
    # Not enough points for extrema
    if len(data) < 3:
        return np.nan

    # Find peaks (maxima)
    peaks, _ = find_peaks(data)
    
    # Find troughs (minima) by negating the data
    troughs, _ = find_peaks(-data)
    
    # Combine and sort extrema (peaks + troughs)
    extrema = np.sort(np.concatenate([peaks, troughs]))

    # If fewer than two extrema, return NaN
    if len(extrema) < 2:
        return np.nan

    # Calculate absolute amplitude changes between consecutive extrema
    amplitude_changes = np.abs(np.diff(data[extrema]))

    print("Peaks:", peaks)
    print("Troughs:", troughs)
    print("Extrema:", extrema)
    print("Amplitude changes:", amplitude_changes)

    # Return the mean of amplitude changes
    return np.mean(amplitude_changes) if len(amplitude_changes) > 0 else np.nan

