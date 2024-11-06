import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_mean import calculate_mean

def test_calculate_mean():
    """
    Test that calculate_mean correctly calculates the mean value of a time series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_mean(data) == 3.0, "The mean value should be 3.0"

def test_calculate_mean_empty_series():
    """
    Test that calculate_mean returns NaN for an empty time series.
    """
    data = pd.Series([])
    assert pd.isna(calculate_mean(data)), "The mean of an empty series should be NaN"

def test_calculate_mean_numpy_array():
    """
    Test that calculate_mean correctly calculates the mean value of a numpy array.
    """
    data = np.array([1, 2, 3, 4, 5])
    assert calculate_mean(data) == 3.0, "The mean value should be 3.0 for numpy array"

def test_calculate_mean_negative_values():
    """
    Test that calculate_mean correctly calculates the mean for a series with negative values.
    """
    data = pd.Series([-1, -2, -3, -4, -5])
    assert calculate_mean(data) == -3.0, "The mean value should be -3.0"
