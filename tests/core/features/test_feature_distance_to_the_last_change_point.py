import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_distance_to_the_last_change_point import calculate_distance_to_last_trend_change

# Test a basic case with one trend change
def test_trend_change_basic_case():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
    result = calculate_distance_to_last_trend_change(data, window_size=3)
    assert result == 4, f"Expected 4, got {result}"

# Test monotonic increasing series (no trend change expected)
def test_trend_change_monotonic_increasing():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_distance_to_last_trend_change(data, window_size=3)
    assert result is None, "Expected None for monotonic increasing series"

# Test monotonic decreasing series (no trend change expected)
def test_trend_change_monotonic_decreasing():
    data = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    result = calculate_distance_to_last_trend_change(data, window_size=3)
    assert result is None, "Expected None for monotonic decreasing series"


# Test series with no trend changes
def test_trend_change_no_change_detected():
    data = pd.Series([1, 1, 1, 1, 1, 1])
    result = calculate_distance_to_last_trend_change(data, window_size=3)
    assert result is None, "Expected None when no trend change is detected"


# Test invalid (negative) window size
def test_trend_change_insufficient_window_size():
    data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Window size must be a positive integer."):
        calculate_distance_to_last_trend_change(data, window_size=-5)


# Test a single trend change
def test_trend_change_single_trend_change():
    data = pd.Series([1, 2, 3, 4, 3, 2, 1])
    result = calculate_distance_to_last_trend_change(data, window_size=3)
    assert result == 1, f"Expected 1, got {result}"

# Test a constant series with no trend changes
def test_trend_change_constant_series():
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_distance_to_last_trend_change(data, window_size=3)
    assert result is None, "Expected None for constant series"
