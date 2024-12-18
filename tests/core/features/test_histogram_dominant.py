import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.histogram_dominant import calculate_dominant

# Test for a basic case with a dominant value
def test_dominant_basic_case():
    data = pd.Series([1, 1, 2, 3, 3, 3, 4, 5])
    result = calculate_dominant(data)
    expected = 3.0
    assert result == expected, f"Expected {expected}, got {result}"

# Test for multiple modes with the same frequency
def test_dominant_multiple_modes():
    data = pd.Series([1, 1, 2, 2, 3, 3])
    result = calculate_dominant(data)
    expected = 1.0  # The first bin with the highest frequency is returned
    assert result == expected, f"Expected {expected}, got {result}"

# Test for a series with a single value
def test_dominant_single_value():
    data = pd.Series([10])
    result = calculate_dominant(data)
    expected = 10.0
    assert result == expected, f"Expected {expected}, got {result}"

# Test for a series where all values are unique
def test_dominant_all_unique():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_dominant(data)
    expected = 1.0  # The first bin is returned
    assert result == expected, f"Expected {expected}, got {result}"

# Test for an empty series
def test_dominant_empty_series():
    data = pd.Series([], dtype=float)
    result = calculate_dominant(data)
    assert np.isnan(result), f"Expected NaN, got {result}"

# Test for a series containing NaN values
def test_dominant_with_nan():
    data = pd.Series([1, 2, np.nan, 3, 3, 3])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_dominant(data)

# Test for numpy array input
def test_dominant_numpy_array():
    data = np.array([1, 1, 2, 3, 3, 3, 4, 5])
    result = calculate_dominant(data)
    expected = 3.0
    assert result == expected, f"Expected {expected}, got {result}"

# Test for returning the center of the dominant bin
def test_dominant_bin_center():
    data = pd.Series([1, 1, 2, 2, 3, 3])
    result = calculate_dominant(data, bins=3, return_bin_center=True)
    expected = 1.3333333333333333  # Center of the dominant bin
    assert result == expected, f"Expected {expected}, got {result}"

# Test for custom number of bins
def test_dominant_custom_bins():
    data = pd.Series([1, 2, 2, 3, 4, 4, 5, 5, 5])
    result = calculate_dominant(data, bins=3)
    expected = 3.6666666666666665  # Lower bound of the dominant bin
    assert result == expected, f"Expected {expected}, got {result}"
