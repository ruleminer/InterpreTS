import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_mean import calculate_mean

def test_calculate_mean_simple_series():
    """Test mean calculation for a simple series."""
    data = pd.Series([1, 2, 3, 4, 5])
    expected = 3.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_mean_empty_series():
    """Test mean calculation for an empty series."""
    data = pd.Series([], dtype=float)
    result = calculate_mean(data)
    assert np.isnan(result), "Mean of an empty series should be NaN."

def test_calculate_mean_single_value():
    """Test mean calculation for a series with a single value."""
    data = pd.Series([42])
    expected = 42.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_mean_negative_values():
    """Test mean calculation for a series with negative values."""
    data = pd.Series([-1, -2, -3, -4, -5])
    expected = -3.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_mean_mixed_values():
    """Test mean calculation for a series with mixed positive and negative values."""
    data = pd.Series([-1, -2, 3, 4, 5])
    expected = 1.8
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_mean_with_nan():
    """Test mean calculation for a series containing NaN values."""
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError):
        calculate_mean(data)

def test_calculate_mean_numpy_array():
    """Test mean calculation for a numpy array."""
    data = np.array([1, 2, 3, 4, 5])
    expected = 3.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_mean_constant_series():
    """Test mean calculation for a series with constant values."""
    data = pd.Series([10, 10, 10, 10])
    expected = 10.0
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_mean_large_numbers():
    """Test mean calculation for a series with very large numbers."""
    data = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    expected = 3e10
    result = calculate_mean(data)
    assert result == expected, f"Expected {expected}, but got {result}."

def test_calculate_mean_small_numbers():
    """Test mean calculation for a series with very small numbers."""
    data = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    expected = 3e-10
    result = calculate_mean(data)
    assert result == pytest.approx(expected, rel=1e-9), f"Expected {expected}, but got {result}."
