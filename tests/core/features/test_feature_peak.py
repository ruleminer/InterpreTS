import pytest
import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data
from interpreTS.core.features.feature_peak import calculate_peak  # Adjust module path as necessary

# Test peak calculation for the full series
def test_calculate_peak_full_series():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data)
    assert result == 7, "Peak calculation failed for the full series"

# Test peak calculation within a specified range
def test_calculate_peak_with_start_and_end():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data, start=1, end=4)
    assert result == 5, "Peak calculation failed for specified range"

# Test peak calculation with only the start index specified
def test_calculate_peak_with_start_only():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data, start=2)
    assert result == 7, "Peak calculation failed when only start is specified"

# Test peak calculation with only the end index specified
def test_calculate_peak_with_end_only():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data, end=3)
    assert result == 5, "Peak calculation failed when only end is specified"

# Test peak calculation for an empty series
def test_calculate_peak_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_peak(data)

# Test peak calculation for a single value
def test_calculate_peak_single_value():
    data = pd.Series([42])
    result = calculate_peak(data)
    assert result == 42, "Peak calculation failed for a single value"

# Test peak calculation for out-of-bounds start and end indices
def test_calculate_peak_start_end_out_of_bounds():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_peak(data, start=10, end=15)
    assert np.isnan(result), "Peak calculation should return NaN for out-of-bounds range"

# Test peak calculation when start index is greater than end index
def test_calculate_peak_start_greater_than_end():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_peak(data, start=4, end=2)
    assert np.isnan(result), "Peak calculation should return NaN when start > end"

# Test peak calculation for a numpy array input
def test_calculate_peak_numpy_array():
    data = np.array([1, 3, 5, 2, 7])
    result = calculate_peak(data)
    assert result == 7, "Peak calculation failed for numpy array"

# Test peak calculation for a large dataset
def test_calculate_peak_large_dataset():
    data = pd.Series(range(1000000))
    result = calculate_peak(data)
    assert result == 999999, "Peak calculation failed for a large dataset"
