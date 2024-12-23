import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_amplitude_change_rate import calculate_amplitude_change_rate  

# Test amplitude change rate for a simple series with clear peaks and troughs
def test_amplitude_change_rate_simple_series():
    data = pd.Series([1, 5, 2, 8, 3])
    result = calculate_amplitude_change_rate(data)
    expected = (abs(5 - 2) + abs(2 - 8)) / 2  # Average of amplitude changes
    assert result == expected, f"Expected {expected}, but got {result}"

# Test amplitude change rate for a flat series
def test_amplitude_change_rate_flat_series():
    data = pd.Series([4, 4, 4, 4])
    result = calculate_amplitude_change_rate(data)
    assert np.isnan(result), "Amplitude change rate should return NaN for flat series"

# Test amplitude change rate for a single peak and trough
def test_amplitude_change_rate_single_peak_trough():
    data = pd.Series([3, 7, 3])
    result = calculate_amplitude_change_rate(data)
    expected = np.nan
    assert np.isnan(result)

# Test amplitude change rate for no peaks or troughs
def test_amplitude_change_rate_no_extrema():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_amplitude_change_rate(data)
    assert np.isnan(result), "Amplitude change rate should return NaN if no extrema are found"

# Test amplitude change rate for series with NaN values
def test_amplitude_change_rate_with_nan_values():
    data = pd.Series([1, np.nan, 2, 3])
    with pytest.raises(ValueError, match="Input data contains NaN values."):
        calculate_amplitude_change_rate(data)

# Test amplitude change rate for a large dataset
def test_amplitude_change_rate_large_dataset():
    data = pd.Series(np.sin(np.linspace(0, 10 * np.pi, 10000)))  # Sinusoidal wave
    result = calculate_amplitude_change_rate(data)
    assert result > 0, "Amplitude change rate should be positive for sinusoidal data"

# Test amplitude change rate for invalid data types
def test_amplitude_change_rate_invalid_data():
    data = ["a", "b", "c"]
    with pytest.raises(TypeError, match="Data must be a pd.Series or np.ndarray."):
        calculate_amplitude_change_rate(data)

# Test amplitude change rate for multidimensional data
def test_amplitude_change_rate_multidimensional_data():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Data must be one-dimensional."):
        calculate_amplitude_change_rate(data)
