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

# Test for a time series containing negative values
def test_mean_change_with_negative_values():
    data = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    result = calculate_mean_change(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)
