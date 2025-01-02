import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_trough import calculate_trough  

# Test trough calculation for the full series
def test_calculate_trough_full_series():
    data = pd.Series([1, 3, 5, 2, -2])
    result = calculate_trough(data)
    assert result == -2, "Trough calculation failed for the full series"

# Test trough calculation within a specified range
def test_calculate_trough_with_start_and_end():
    data = pd.Series([1, 3, 5, 2, -2])
    result = calculate_trough(data, start=1, end=4)
    assert result == 2, "Trough calculation failed for specified range"

# Test trough calculation with only the start index specified
def test_calculate_trough_with_start_only():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_trough(data, start=2)
    assert result == 2, "Trough calculation failed when only start is specified"

# Test trough calculation with only the end index specified
def test_calculate_trough_with_end_only():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_trough(data, end=3)
    assert result == 1, "Trough calculation failed when only end is specified"

# Test trough calculation for a single value
def test_calculate_trough_single_value():
    data = pd.Series([42])
    result = calculate_trough(data)
    assert result == 42, "Trough calculation failed for a single value"

# Test trough calculation for out-of-bounds start and end indices
def test_calculate_trough_start_end_out_of_bounds():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_trough(data, start=10, end=15)
    assert np.isnan(result), "Trough calculation should return NaN for out-of-bounds range"

# Test trough calculation when start index is greater than end index
def test_calculate_trough_start_greater_than_end():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_trough(data, start=4, end=2)
    assert np.isnan(result), "Trough calculation should return NaN when start > end"

# Test trough calculation for a numpy array input
def test_calculate_trough_numpy_array():
    data = np.array([1, 3, 5, 2, 7])
    result = calculate_trough(data)
    assert result == 1, "Trough calculation failed for numpy array"

# Test trough calculation for a large dataset
def test_calculate_trough_large_dataset():
    data = pd.Series(range(1000000, 0, -1))
    result = calculate_trough(data)
    assert result == 1, "Trough calculation failed for a large dataset"
