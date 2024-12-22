import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_length import calculate_length

# Test length calculation for a basic series
def test_calculate_length_basic():
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 5
    result = calculate_length(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test length calculation for an empty series
def test_calculate_length_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_length(data)

# Test length calculation for a large series
def test_calculate_length_large_series():
    data = pd.Series(range(1_000_000))
    expected = 1_000_000
    result = calculate_length(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test length calculation for a series with NaN values
def test_calculate_length_nan_values():
    data = pd.Series([1, np.nan, 3, np.nan, 5])
    expected = 5  # Length counts all elements, including NaN
    result = calculate_length(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test length calculation for a numpy array
def test_calculate_length_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    expected = 5
    result = calculate_length(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test length calculation for invalid input types
def test_calculate_length_invalid_type():
    with pytest.raises(TypeError, match="Data must be a pandas Series, DataFrame, or numpy array."):
        calculate_length("invalid_type")

# Test length calculation for a multidimensional array
def test_calculate_length_multidimensional_array():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError, match="Data must be one-dimensional."):
        calculate_length(data)
