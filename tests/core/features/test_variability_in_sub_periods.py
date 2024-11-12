import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.variability_in_sub_periods import calculate_variability_in_sub_periods

def test_calculate_variability_in_sub_periods_strong_variability():
    """
    Test that calculate_variability_in_sub_periods identifies strong variability
    in a time series with large fluctuations.
    """
    data = pd.Series([1, 10, 1, 10, 1, 10, 1, 10])
    result = calculate_variability_in_sub_periods(data, window_size=4)
    assert all(val > 10 for val in result), "The variability should be high for data with strong fluctuations"

def test_calculate_variability_in_sub_periods_weak_variability():
    """
    Test that calculate_variability_in_sub_periods identifies weak variability
    in a time series with small fluctuations.
    """
    data = pd.Series([1, 1.1, 1.2, 1.1, 1, 1.05, 1.1, 1.15])
    result = calculate_variability_in_sub_periods(data, window_size=4)
    assert all(val < 0.1 for val in result), "The variability should be low for data with weak fluctuations"

def test_calculate_variability_in_sub_periods_no_variability():
    """
    Test that calculate_variability_in_sub_periods identifies no variability
    in a time series with constant values.
    """
    data = pd.Series([5, 5, 5, 5, 5, 5, 5, 5])
    result = calculate_variability_in_sub_periods(data, window_size=4)
    assert all(val == 0 for val in result), "The variability should be zero for constant data"

def test_calculate_variability_in_sub_periods_empty_series():
    """
    Test that calculate_variability_in_sub_periods returns an empty list for an empty series.
    """
    data = pd.Series([])
    result = calculate_variability_in_sub_periods(data, window_size=4)
    assert result.empty, "The result should be empty for an empty series"

def test_calculate_variability_in_sub_periods_insufficient_data():
    """
    Test that calculate_variability_in_sub_periods returns an empty result when data
    is insufficient for the specified window size.
    """
    data = pd.Series([5, 6])
    result = calculate_variability_in_sub_periods(data, window_size=4)
    assert result.empty, "The result should be empty for data shorter than the window size"

def test_calculate_variability_in_sub_periods_with_numpy_array():
    """
    Test that calculate_variability_in_sub_periods works correctly with a numpy array.
    """
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_variability_in_sub_periods(data, window_size=5)
    assert len(result) == 2, "The result should have two values for a numpy array with a window size of 5"
    assert all(isinstance(val, float) for val in result), "Each result should be a float (variance value)"

def test_calculate_variability_in_sub_periods_with_overlap():
    """
    Test that calculate_variability_in_sub_periods handles overlapping sub-periods correctly.
    """
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_variability_in_sub_periods(data, window_size=5, step_size=2)
    expected_length = (len(data) - 5) // 2 + 1  # Expected number of overlapping windows
    assert len(result) == expected_length, "Overlap calculation failed"
    assert all(isinstance(v, float) for v in result), "Each result should be a float (variance value)"

def test_calculate_variability_in_sub_periods_high_variance():
    """
    Test that calculate_variability_in_sub_periods identifies high variance in sub-periods
    with data that fluctuates greatly within each window.
    """
    data = pd.Series([1, 100, 1, 100, 1, 100, 1, 100, 1, 100])
    result = calculate_variability_in_sub_periods(data, window_size=3)
    assert all(val > 1000 for val in result), "The variability should be high for high-fluctuation data"

def test_calculate_variability_in_sub_periods_with_small_window_size():
    """
    Test that calculate_variability_in_sub_periods correctly calculates variability with a small window size.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_variability_in_sub_periods(data, window_size=2)
    assert len(result) == len(data) - 1, "The number of results should be data length - 1 with a window size of 2"
    assert result.iloc[0] == 0.5, "The variance of the first window should be 0.5"