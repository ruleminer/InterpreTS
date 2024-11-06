import pytest
import pandas as pd
import numpy as np
from src.core.features.histogram_dominant import calculate_dominant

def test_calculate_dominant_basic():
    """
    Test that calculate_dominant correctly identifies the dominant value
    in a simple time series.
    """
    data = pd.Series([1, 2, 2, 3, 4, 4, 4, 5])
    assert calculate_dominant(data) == 4.0, "The dominant value should be 4.0"

def test_calculate_dominant_empty_series():
    """
    Test that calculate_dominant returns NaN for an empty time series.
    """
    data = pd.Series([])
    assert pd.isna(calculate_dominant(data)), "The dominant value of an empty series should be NaN"

def test_calculate_dominant_numpy_array():
    """
    Test that calculate_dominant correctly identifies the dominant value in a numpy array.
    """
    data = np.array([1, 1, 2, 3, 3, 3, 4, 5])
    assert calculate_dominant(data) == 3.0, "The dominant value should be 3.0 for the numpy array"

def test_calculate_dominant_multiple_modes():
    """
    Test that calculate_dominant returns one of the most frequent values
    when there are multiple modes in the data.
    """
    data = pd.Series([1, 2, 2, 3, 3, 4])
    dominant_value = calculate_dominant(data, return_bin_center=True)
    assert dominant_value in [2.5, 3.5], "The dominant value should be one of the modes"

def test_calculate_dominant_bins_parameter():
    """
    Test that calculate_dominant respects the 'bins' parameter for histogram granularity.
    """
    data = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    # With fewer bins, the dominant should approximate the most frequent bin
    dominant_value = calculate_dominant(data, bins=3, return_bin_center=True)
    assert dominant_value in [1.5, 4.5], "The dominant value should approximate the largest bin center"

def test_calculate_dominant_negative_values():
    """
    Test that calculate_dominant correctly identifies the dominant value for a series with negative values.
    """
    data = pd.Series([-5, -4, -4, -3, -3, -3, -2, -1])
    assert calculate_dominant(data) == -3.0, "The dominant value should be -3.0"