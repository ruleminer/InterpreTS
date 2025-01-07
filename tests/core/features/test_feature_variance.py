import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_variance import calculate_variance

# Test variance calculation for a basic series
def test_calculate_variance_basic():
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 2.0  # Sample variance
    result = calculate_variance(data)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test variance calculation with population variance
def test_calculate_variance_population():
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 2.0  # Population variance
    result = calculate_variance(data, ddof=0)
    assert result == expected, f"Expected {expected}, but got {result}."

# Test variance calculation for a series with a single value
def test_calculate_variance_single_value():
    data = pd.Series([42])
    expected = 0.0  # Variance of a single value is 0
    result = calculate_variance(data)
    assert result == expected, f"Expected {expected}, but got {result}."


# Test variance calculation for a series with very small numbers
def test_calculate_variance_small_numbers():
    data = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    expected = 2.5e-20  # Sample variance
    result = calculate_variance(data)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}."

# Test variance calculation for a series with all identical values
def test_calculate_variance_all_identical_values():
    data = pd.Series([3, 3, 3, 3, 3])
    expected = 0.0  # Variance of identical values is 0
    result = calculate_variance(data)
    assert result == expected, f"Expected {expected}, but got {result}."
