import pytest
import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data
from interpreTS.core.features.feature_peak import calculate_peak  # Zmień "your_module" na odpowiedni moduł


def test_calculate_peak_full_series():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data)
    assert result == 7, "Peak calculation failed for the full series"


def test_calculate_peak_with_start_and_end():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data, start=1, end=4)
    assert result == 5, "Peak calculation failed for specified range"


def test_calculate_peak_with_start_only():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data, start=2)
    assert result == 7, "Peak calculation failed when only start is specified"


def test_calculate_peak_with_end_only():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_peak(data, end=3)
    assert result == 5, "Peak calculation failed when only end is specified"


def test_calculate_peak_empty_series():
    data = pd.Series([])
    result = calculate_peak(data)
    assert np.isnan(result), "Peak calculation should return NaN for an empty series"


def test_calculate_peak_single_value():
    data = pd.Series([42])
    result = calculate_peak(data)
    assert result == 42, "Peak calculation failed for a single value"


def test_calculate_peak_start_end_out_of_bounds():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_peak(data, start=10, end=15)
    assert np.isnan(result), "Peak calculation should return NaN for out-of-bounds range"


def test_calculate_peak_start_greater_than_end():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_peak(data, start=4, end=2)
    assert np.isnan(result), "Peak calculation should return NaN when start > end"


def test_calculate_peak_numpy_array():
    data = np.array([1, 3, 5, 2, 7])
    result = calculate_peak(data)
    assert result == 7, "Peak calculation failed for numpy array"


def test_calculate_peak_large_dataset():
    data = pd.Series(range(1000000))
    result = calculate_peak(data)
    assert result == 999999, "Peak calculation failed for a large dataset"