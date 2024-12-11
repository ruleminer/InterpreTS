import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_entropy import calculate_entropy

def test_calculate_entropy_constant_data():
    """Test entropy calculation for constant data."""
    data = pd.Series([5, 5, 5, 5])
    result = calculate_entropy(data)
    assert result == 0.0, f"Expected entropy 0.0 for constant data. Got: {result}"

def test_calculate_entropy_uniform_data():
    """Test entropy calculation for uniformly distributed data."""
    data = pd.Series([1, 2, 3, 4])
    result = calculate_entropy(data)
    assert result == 1.0, f"Expected entropy 1.0 for uniformly distributed data. Got: {result}"

def test_calculate_entropy_short_data():
    """Test entropy calculation for data that is too short."""
    data = pd.Series([1])
    with pytest.raises(ValueError, match="Input data is too short to calculate entropy."):
        calculate_entropy(data)

def test_calculate_entropy_invalid_bins():
    """Test entropy calculation with invalid bin count."""
    data = pd.Series([1, 2, 3, 4])
    with pytest.raises(ValueError, match="Number of bins must be at least 2."):
        calculate_entropy(data, bins=1)

def test_calculate_entropy_with_nan():
    """Test entropy calculation for data with NaN values."""
    data = pd.Series([1, 2, np.nan, 4])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_entropy(data)

def test_calculate_entropy_non_numeric_data():
    """Test entropy calculation for non-numeric data."""
    data = pd.Series(["a", "b", "c", "a"])
    with pytest.raises(TypeError, match="Data must be numeric."):
        calculate_entropy(data)

def test_calculate_entropy_real_numbers():
    """Test entropy calculation for a series with real numbers."""
    data = pd.Series([1.1, 1.2, 1.3, 1.4, 1.5])
    result = calculate_entropy(data)
    assert 0.0 <= result <= 1.0, f"Expected entropy in range [0.0, 1.0]. Got: {result}"

def test_calculate_entropy_duplicate_values():
    """Test entropy calculation for data with duplicate values."""
    data = pd.Series([1, 1, 2, 2, 3, 3, 3])
    result = calculate_entropy(data)
    assert 0.0 <= result <= 1.0, f"Expected entropy in range [0.0, 1.0]. Got: {result}"

def test_calculate_entropy_edge_case_bins():
    """Test entropy calculation with a number of bins equal to unique values."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_entropy(data, bins=len(data))
    assert 0.0 <= result <= 1.0, f"Expected entropy in range [0.0, 1.0]. Got: {result}"

def test_calculate_entropy_large_data():
    """Test entropy calculation for a large dataset."""
    data = pd.Series(np.random.rand(1000))
    result = calculate_entropy(data)
    assert 0.0 <= result <= 1.0, f"Expected entropy in range [0.0, 1.0]. Got: {result}"
