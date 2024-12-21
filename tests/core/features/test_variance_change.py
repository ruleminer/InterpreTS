import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.variance_change import calculate_change_in_variance

# Test for basic functionality with evenly spaced data
def test_change_in_variance_basic_case():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_change_in_variance(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)

# Test for data containing NaN values
def test_change_in_variance_with_nan_values():
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_change_in_variance(data, window_size=3)

# Test for an empty series
def test_change_in_variance_empty_series():
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_change_in_variance(data, window_size=3)

# Test for insufficient data for the specified window size
def test_change_in_variance_insufficient_data():
    data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="The time series is too short for the specified rolling window size."):
        calculate_change_in_variance(data, window_size=5)

# Test for data with a single repeated value
def test_change_in_variance_single_value():
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_change_in_variance(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)

# Test for invalid window size
def test_change_in_variance_invalid_window_size():
    data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Window size must be a positive integer."):
        calculate_change_in_variance(data, window_size=-3)

# Test for data provided as a numpy array
def test_change_in_variance_numpy_array():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_change_in_variance(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)

# Test for data containing extreme values
def test_change_in_variance_with_extreme_values():
    data = pd.Series([1e9, 1e9, 1e9, -1e9, -1e9, -1e9, 0, 0, 0, 0])
    result = calculate_change_in_variance(data, window_size=3)
    assert not result.isnull().all(), "Expected valid results for extreme values"

# Test for linearly increasing data
def test_change_in_variance_linear_data():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_change_in_variance(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)

# Test for window size equal to data length
def test_change_in_variance_window_size_equals_data_length():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_change_in_variance(data, window_size=5)
    expected = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    pd.testing.assert_series_equal(result, expected)

# Test for very large window size
def test_change_in_variance_large_window_size():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError, match="The time series is too short for the specified rolling window size."):
        calculate_change_in_variance(data, window_size=15)

# Test for a series with all identical values
def test_change_in_variance_all_identical_values():
    data = pd.Series([100] * 10)
    result = calculate_change_in_variance(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)
