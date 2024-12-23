import pandas as pd
import numpy as np
from interpreTS.core.features.feature_peak import calculate_peak

def calculate_amplitude_change_rate(data):
    """
    Calculate the average amplitude change rate in a time series,
    defined as the mean change in amplitude between consecutive peaks (maxima-minima).

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the amplitude change rate is to be calculated.

    Returns
    -------
    float
        The average amplitude change rate. Returns NaN if no peaks are found or if data is invalid.

    Raises
    ------
    TypeError
        If the data is not numeric or not a valid type.
    ValueError
        If the data contains NaN values or is not one-dimensional.
    """
    # Validate input data
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Data must be one-dimensional.")
        data = data.squeeze()
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError("Data must be a pd.Series or np.ndarray.")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if isinstance(data, np.ndarray) and data.ndim != 1:
        raise ValueError("Data must be one-dimensional.")

    # Identify peaks (local maxima) and troughs (local minima)
    peaks, troughs = [], []
    for i in range(1, len(data) - 1):
        if data[i - 1] < data[i] > data[i + 1]:  # Peak condition
            peaks.append(i)
        elif data[i - 1] > data[i] < data[i + 1]:  # Trough condition
            troughs.append(i)

    # Combine and sort extrema indices
    extrema = sorted(peaks + troughs)
    if len(extrema) < 2:
        return np.nan  # Not enough extrema to calculate changes

    # Calculate amplitude changes between consecutive extrema
    amplitude_changes = []
    for i in range(len(extrema) - 1):
        max_value = calculate_peak(data, start=extrema[i], end=extrema[i + 1])
        min_value = calculate_peak(-data, start=extrema[i], end=extrema[i + 1])
        amplitude_changes.append(abs(max_value - min_value))

    # Return the mean of amplitude changes
    return np.mean(amplitude_changes) if amplitude_changes else np.nan
