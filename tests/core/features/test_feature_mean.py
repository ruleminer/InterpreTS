import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_mean import calculate_mean

# Test mean calculation for a simple series
def test_calculate_mean_simple_series():
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 3.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test mean calculation for an empty series
def test_calculate_mean_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_mean(data)

# Test mean calculation for a series with a single value
def test_calculate_mean_single_value():
    data = pd.Series([42])
    expected = 42.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test mean calculation for a series with negative values
def test_calculate_mean_negative_values():
    data = pd.Series([-1, -2, -3, -4, -5])
    expected = -3.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test mean calculation for a series with mixed positive and negative values
def test_calculate_mean_mixed_values():
    data = pd.Series([-1, -2, 3, 4, 5])
    expected = 1.8
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test mean calculation for a series containing NaN values
def test_calculate_mean_with_nan():
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_mean(data)

# Test mean calculation for a numpy array input
def test_calculate_mean_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    expected = 3.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test mean calculation for a series with constant values
def test_calculate_mean_constant_series():
    data = pd.Series([10, 10, 10, 10])
    expected = 10.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test mean calculation for a series with very large numbers
def test_calculate_mean_large_numbers():
    data = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    expected = 3e10
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test mean calculation for a series with very small numbers
def test_calculate_mean_small_numbers():
    data = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    expected = 3e-10
    result = calculate_mean(data)
    assert result == pytest.approx(expected, rel=1e-9), f"Expected {expected}, but got {result}."
