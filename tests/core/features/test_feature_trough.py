import pytest
import pandas as pd
import numpy as np
from src.core.features.feature_trough import calculate_trough

def test_calculate_trough():
    """
    Test that calculate_trough correctly calculates the trough (minimum) value of a time series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_trough(data) == 1.0, "The trough value should be 1.0"

def test_calculate_trough_empty_series():
    """
    Test that calculate_trough returns NaN for an empty time series.
    """
    data = pd.Series([])
    assert pd.isna(calculate_trough(data)), "The trough of an empty series should be NaN"

def test_calculate_trough_numpy_array():
    """
    Test that calculate_trough correctly calculates the trough (minimum) value of a numpy array.
    """
    data = np.array([1, 2, 3, 4, 5])
    assert calculate_trough(data) == 1.0, "The trough value should be 1.0 for numpy array"

def test_calculate_trough_with_negative_values():
    """
    Test that calculate_trough correctly calculates the trough (minimum) for a series with negative values.
    """
    data = pd.Series([-1, -2, -3, -4, -5])
    assert calculate_trough(data) == -5.0, "The trough value should be -5.0"

def test_calculate_trough_with_start_and_end():
    """
    Test that calculate_trough correctly calculates the trough within a specified start and end range.
    """
    data = pd.Series([1, 2, 5, 4, 3])
    assert calculate_trough(data, start=1, end=4) == 2.0, "The trough value within index range 1 to 4 should be 2.0"

def test_calculate_trough_with_out_of_range_indices():
    """
    Test that calculate_trough handles out-of-range start and end indices gracefully.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert np.isnan(calculate_trough(data, start=10)), "The trough with out-of-range start index should be NaN"
    assert np.isnan(calculate_trough(data, end=-10)), "The trough with out-of-range end index should be NaN"

def test_calculate_trough_with_start_greater_than_end():
    """
    Test that calculate_trough returns NaN if start index is greater than end index.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert pd.isna(calculate_trough(data, start=4, end=2)), "The trough should be NaN if start index is greater than end index"
