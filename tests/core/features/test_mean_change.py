import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.mean_change import calculate_mean_change

# Test for a valid time series input
def test_mean_change_valid_series():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_mean_change(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)

# Test for a valid NumPy array input
def test_mean_change_valid_array():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_mean_change(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)

# Test for a time series that is too short for the rolling window
def test_mean_change_insufficient_data():
    data = pd.Series([1, 2])
    with pytest.raises(ValueError, match="The time series is too short for the specified rolling window size."):
        calculate_mean_change(data, window_size=3)

# Test for data containing NaN values
def test_mean_change_with_nan():
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_mean_change(data, window_size=3)

# Test for empty data
def test_mean_change_empty_data():
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_mean_change(data, window_size=3)

# Test for a time series with constant values
def test_mean_change_constant_values():
    data = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    result = calculate_mean_change(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)

# Test for a very small rolling window size
def test_mean_change_small_window():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_mean_change(data, window_size=1)
    expected = pd.Series([np.nan, 1.0, 1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)

# Test for a non-integer rolling window size
def test_mean_change_invalid_window_size():
    data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Window size must be a positive integer."):
        calculate_mean_change(data, window_size=-3)

# Test for a time series containing negative values
def test_mean_change_with_negative_values():
    data = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    result = calculate_mean_change(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)
