import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_peak import calculate_peak

def test_calculate_peak():
    """
    Test that calculate_peak correctly calculates the peak (maximum) value of a time series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_peak(data) == 5.0, "The peak value should be 5.0"

def test_calculate_peak_empty_series():
    """
    Test that calculate_peak returns NaN for an empty time series.
    """
    data = pd.Series([])
    assert pd.isna(calculate_peak(data)), "The peak of an empty series should be NaN"

def test_calculate_peak_numpy_array():
    """
    Test that calculate_peak correctly calculates the peak (maximum) value of a numpy array.
    """
    data = np.array([1, 2, 3, 4, 5])
    assert calculate_peak(data) == 5.0, "The peak value should be 5.0 for numpy array"

def test_calculate_peak_with_negative_values():
    """
    Test that calculate_peak correctly calculates the peak (maximum) for a series with negative values.
    """
    data = pd.Series([-1, -2, -3, -4, -5])
    assert calculate_peak(data) == -1.0, "The peak value should be -1.0"

def test_calculate_peak_with_start_and_end():
    """
    Test that calculate_peak correctly calculates the peak within a specified start and end range.
    """
    data = pd.Series([1, 2, 5, 4, 3])
    assert calculate_peak(data, start=1, end=4) == 5.0, "The peak value within index range 1 to 4 should be 5.0"

def test_calculate_peak_with_out_of_range_indices():
    """
    Test that calculate_peak handles out-of-range start and end indices gracefully.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert np.isnan(calculate_peak(data, start=10)), "The peak with out-of-range start index should be NaN"
    assert np.isnan(calculate_peak(data, end=-10)), "The peak with out-of-range end index should be NaN"

def test_calculate_peak_with_start_greater_than_end():
    """
    Test that calculate_peak returns NaN if start index is greater than end index.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert pd.isna(calculate_peak(data, start=4, end=2)), "The peak should be NaN if start index is greater than end index"
