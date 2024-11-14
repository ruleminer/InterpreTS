import pytest
import pandas as pd
import numpy as np
from src.core.features.feature_bizarre_mean import calculate_bizarre_mean

def test_calculate_bizarre_mean():
    data = pd.Series([1, 2, 3, 4, 5])
    expected_result = 3.0 * np.log(5)
    assert calculate_bizarre_mean(data) == expected_result, "The bizarre mean should be mean(data) * log(len(data))"

def test_calculate_bizarre_mean_empty_series():
    data = pd.Series([])
    assert pd.isna(calculate_bizarre_mean(data)), "The bizarre mean of an empty series should be NaN"

def test_calculate_bizarre_mean_with_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    expected_result = 3.0 * np.log(5)
    assert calculate_bizarre_mean(pd.Series(data)) == expected_result, "The bizarre mean should be mean(data) * log(len(data))"

def test_calculate_bizarre_mean_negative_values():
    data = pd.Series([-1, -2, -3, -4, -5])
    expected_result = -3.0 * np.log(5)
    assert calculate_bizarre_mean(data) == expected_result, "The bizarre mean should correctly handle negative values"
