import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_binarize_mean import calculate_binarize_mean

def test_calculate_binarize_mean():
    data = pd.Series([1, 2, 3, 4, 5])
    expected_result = 0.4  # 3 out of 5 elements are above the mean (3)
    assert calculate_binarize_mean(data) == expected_result, "The binarize mean should be 0.6 for this series"

def test_calculate_binarize_mean_empty_series():
    data = pd.Series([])
    assert pd.isna(calculate_binarize_mean(data)), "The binarize mean of an empty series should be NaN"

def test_calculate_binarize_mean_with_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    expected_result = 0.4
    assert calculate_binarize_mean(pd.Series(data)) == expected_result, "The binarize mean should be 0.6 for this numpy array"

def test_calculate_binarize_mean_negative_values():
    data = pd.Series([-1, -2, -3, -4, -5])
    expected_result = 0.4  # 2 out of 5 elements are above the mean (-3)
    assert calculate_binarize_mean(data) == expected_result, "The binarize mean should correctly handle negative values"
