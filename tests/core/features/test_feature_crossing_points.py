import pytest
import numpy as np
import pandas as pd
from interpreTS.core.features.feature_crossing_points import calculate_crossing_points

# Test basic case with multiple mean crossings
def test_crossing_points_basic_case():
    data = pd.Series([1, -1, 2, -2, 3, -3])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 5, 'crossing_points': [0, 1, 2, 3, 4]}
    assert result == expected, f"Expected {expected}, got {result}"

# Test when all data points are above the mean
def test_crossing_points_no_crossings_all_above():
    data = pd.Series([2, 3, 4, 5])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 1, 'crossing_points': [1]}
    assert result == expected, f"Expected {expected}, got {result}"

# Test when all data points are below the mean
def test_crossing_points_no_crossings_all_below():
    data = pd.Series([-5, -4, -3, -2])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 1, 'crossing_points': [1]}
    assert result == expected, f"Expected {expected}, got {result}"

# Test a single crossing of the mean
def test_crossing_points_single_crossing():
    data = pd.Series([1, -1])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 1, 'crossing_points': [0]}
    assert result == expected, f"Expected {expected}, got {result}"

# Test alternating positive and negative values
def test_crossing_points_alternating_data():
    data = pd.Series([1, -1, 1, -1, 1, -1])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 5, 'crossing_points': [0, 1, 2, 3, 4]}
    assert result == expected, f"Expected {expected}, got {result}"

# Test with empty data
def test_crossing_points_empty_data():
    data = pd.Series([], dtype=float)
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 0, 'crossing_points': []}
    assert result == expected, f"Expected {expected}, got {result}"

# Test constant data where all points equal the mean
def test_crossing_points_constant_data():
    data = pd.Series([3, 3, 3, 3])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 0, 'crossing_points': []}
    assert result == expected, f"Expected {expected}, got {result}"

# Test with NaN values in the data
def test_crossing_points_with_nan():
    data = pd.Series([1, np.nan, -1, 2])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_crossing_points(data)

# Test functionality with a numpy array input
def test_crossing_points_numpy_array():
    data = np.array([1, -1, 1, -1, 1])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 4, 'crossing_points': [0, 1, 2, 3]}
    assert result == expected, f"Expected {expected}, got {result}"

# Test negative-only data with mean crossings
def test_crossing_points_negative_data():
    data = pd.Series([-3, -1, -4, -2, -5])
    result = calculate_crossing_points(data)
    expected = {'crossing_count': 4, 'crossing_points': [0, 1, 2, 3]}
    assert result == expected, f"Expected {expected}, got {result}"
