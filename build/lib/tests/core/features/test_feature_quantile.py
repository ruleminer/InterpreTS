import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_quantile import calculate_quantile

# Test quantile calculation for a basic series
def test_calculate_quantile_basic():
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 3.0  # Median (50th percentile)
    result = calculate_quantile(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test quantile calculation for a specific percentile
def test_calculate_quantile_specific():
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 4.0  # 75th percentile
    result = calculate_quantile(data, quantile=0.75)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test quantile calculation for a series with a single value
def test_calculate_quantile_single_value():
    data = pd.Series([42])
    expected = 42.0  # Any quantile of a single value is the value itself
    result = calculate_quantile(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test quantile calculation for a series containing NaN values
def test_calculate_quantile_with_nan():
    data = pd.Series([1, np.nan, 3, 4])
    with pytest.raises(ValueError, match="Input data contains NaN values."):
        calculate_quantile(data)

# Test quantile calculation for non-numeric data
def test_calculate_quantile_non_numeric_data():
    data = pd.Series(['a', 'b', 'c'])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        calculate_quantile(data)

# Test quantile calculation for a multidimensional array
def test_calculate_quantile_multidimensional_array():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError, match="Data must be one-dimensional."):
        calculate_quantile(data)

# Test quantile calculation for all identical values
def test_calculate_quantile_all_identical_values():
    data = pd.Series([3, 3, 3, 3, 3])
    expected = 3.0
    result = calculate_quantile(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test quantile calculation for a series with large numbers
def test_calculate_quantile_large_numbers():
    data = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    expected = 3e10
    result = calculate_quantile(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test quantile calculation for a series with very small numbers
def test_calculate_quantile_small_numbers():
    data = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    expected = 3e-10
    result = calculate_quantile(data)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}."
