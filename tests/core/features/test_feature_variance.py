import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_variance import calculate_variance

def test_calculate_variance_basic():
    """Test variance calculation for a basic series."""
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 2.5  # Sample variance
    result = calculate_variance(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_variance_population():
    """Test variance calculation for a basic series with population variance."""
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 2.0  # Population variance
    result = calculate_variance(data, ddof=0)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_variance_single_value():
    """Test variance calculation for a series with a single value."""
    data = pd.Series([42])
    expected = 0.0  # Variance of a single value is 0
    result = calculate_variance(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_variance_empty_series():
    """Test variance calculation for an empty series."""
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_variance(data)

def test_calculate_variance_with_nan():
    """Test variance calculation for a series containing NaN values."""
    data = pd.Series([1, np.nan, 3, 4])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_variance(data)

def test_calculate_variance_large_numbers():
    """Test variance calculation for a series with large numbers."""
    data = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    expected = 2.5e20  # Sample variance
    result = calculate_variance(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_variance_small_numbers():
    """Test variance calculation for a series with very small numbers."""
    data = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    expected = 2.5e-20  # Sample variance
    result = calculate_variance(data)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}."

def test_calculate_variance_non_numeric_data():
    """Test variance calculation for non-numeric data."""
    data = pd.Series(['a', 'b', 'c'])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        calculate_variance(data)

def test_calculate_variance_multidimensional_array():
    """Test variance calculation for a multidimensional array."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError, match="Data must be one-dimensional."):
        calculate_variance(data)

def test_calculate_variance_all_identical_values():
    """Test variance calculation for a series with all identical values."""
    data = pd.Series([3, 3, 3, 3, 3])
    expected = 0.0  # Variance of identical values is 0
    result = calculate_variance(data)
    assert result == expected, f"Expected {expected}, but got {result}."
