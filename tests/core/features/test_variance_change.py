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

# Test for data with a single repeated value
def test_change_in_variance_single_value():
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_change_in_variance(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)

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

# Test for a series with all identical values
def test_change_in_variance_all_identical_values():
    data = pd.Series([100] * 10)
    result = calculate_change_in_variance(data, window_size=3)
    expected = pd.Series([np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)
