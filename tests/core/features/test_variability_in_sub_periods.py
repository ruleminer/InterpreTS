import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.variability_in_sub_periods import calculate_variability_in_sub_periods

# Test for evenly spaced data with a basic setup
def test_variability_basic_case():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_variability_in_sub_periods(data, window_size=5, step_size=5, ddof=1)
    expected = pd.Series([2.5, 2.5])  # Two non-overlapping windows
    pd.testing.assert_series_equal(result, expected)

# Test for overlapping windows using a custom step size
def test_variability_with_step_size():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_variability_in_sub_periods(data, window_size=5, step_size=3, ddof=1)
    expected = pd.Series([2.5, 2.5])  # Overlapping windows
    pd.testing.assert_series_equal(result, expected)

# Test for non-overlapping windows
def test_variability_non_overlapping_windows():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_variability_in_sub_periods(data, window_size=2, step_size=2)
    expected = pd.Series([0.25, 0.25, 0.25, 0.25, 0.25])  # Five non-overlapping windows
    pd.testing.assert_series_equal(result, expected)

# Test for the degrees of freedom parameter (ddof)
def test_variability_with_ddof():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_variability_in_sub_periods(data, window_size=5, ddof=1)
    expected = pd.Series([2.5])  # Variability of the entire series
    pd.testing.assert_series_equal(result, expected)

# Test for data containing NaN values
def test_variability_with_nan():
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_variability_in_sub_periods(data, window_size=3)

# Test when the window size is greater than the data length
def test_variability_insufficient_data():
    data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Window size must be smaller than or equal to the length of the data."):
        calculate_variability_in_sub_periods(data, window_size=5)

# Test for empty input data
def test_variability_empty_data():
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_variability_in_sub_periods(data, window_size=3)

# Test for data with a single repeated value
def test_variability_single_value():
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_variability_in_sub_periods(data, window_size=3, step_size=1)
    expected = pd.Series([0.0, 0.0, 0.0])  # Variability of overlapping windows
    pd.testing.assert_series_equal(result, expected)

# Test functionality with numpy array input
def test_variability_numpy_input():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_variability_in_sub_periods(data, window_size=5, ddof=0)
    expected = pd.Series([2.0, 2.0])  # Using ddof=0
    pd.testing.assert_series_equal(result, expected)

# Test with a step size larger than the window size
def test_variability_custom_step_size_large():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_variability_in_sub_periods(data, window_size=3, step_size=4)
    expected = pd.Series([0.6666666666666666, 0.6666666666666666])  # Non-overlapping large step
    pd.testing.assert_series_equal(result, expected)
