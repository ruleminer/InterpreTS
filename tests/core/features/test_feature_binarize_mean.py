import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_binarize_mean import calculate_binarize_mean

# Test basic functionality with a simple time series
def test_binarize_mean_basic_case():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_binarize_mean(data)
    expected = 0.6  # 3 values (4, 5) are greater than the mean (3)
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when all values are the same
def test_binarize_mean_all_equal_values():
    data = pd.Series([3, 3, 3, 3, 3])
    result = calculate_binarize_mean(data)
    expected = 0.0  # No value is greater than the mean
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test with a series containing negative values
def test_binarize_mean_with_negative_values():
    data = pd.Series([-5, -3, -1, 1, 3])
    result = calculate_binarize_mean(data)
    expected = 0.6  # 3 values are greater than the mean (0)
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test with a series containing a single value
def test_binarize_mean_single_value():
    data = pd.Series([10])
    result = calculate_binarize_mean(data)
    expected = 1.0  # Single value is always equal to its mean
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test with a series containing NaN values
def test_binarize_mean_with_nan():
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_binarize_mean(data)

# Test with an empty series
def test_binarize_mean_empty_series():
    data = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_binarize_mean(data)

# Test functionality with a numpy array
def test_binarize_mean_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    result = calculate_binarize_mean(pd.Series(data))
    expected = 0.6  # 3 values (4, 5) are greater than the mean (3)
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test with invalid data types
def test_binarize_mean_invalid_data_type():
    data = "invalid data"
    with pytest.raises(TypeError, match="Data must be a pandas Series, DataFrame, or numpy array."):
        calculate_binarize_mean(data)

# Test with a large series of sequential numbers
def test_binarize_mean_large_series():
    data = pd.Series(range(1, 1001))  # Values from 1 to 1000
    result = calculate_binarize_mean(data)
    expected = 0.5  # Exactly half the values are greater than the mean
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"
