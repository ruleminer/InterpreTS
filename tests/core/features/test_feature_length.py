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
