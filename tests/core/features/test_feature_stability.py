import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_stability import calculate_stability

def test_calculate_stability_normal_case():
    """Test stability for a normal time series."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_stability(data)
    assert 0 <= result <= 1, f"Stability should be between 0 and 1. Got: {result}"

def test_calculate_stability_constant_series():
    """Test stability for a constant time series."""
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_stability(data)
    assert result == 1.0, f"Expected stability 1.0 for constant series. Got: {result}"

def test_calculate_stability_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_stability(data)

def test_calculate_stability_high_variance_series():
    """Test stability for a high variance time series."""
    data = pd.Series([1, 100, 50, 75, 200, 150, 300, 250, 400, 350])
    result = calculate_stability(data)
    assert 0 <= result <= 1, f"Stability should be between 0 and 1. Got: {result}"

def test_calculate_stability_non_numeric_data():
    """Test stability for non-numeric data."""
    data = pd.Series(["a", "b", "c", "d"])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        calculate_stability(data)

def test_calculate_stability_with_nan_values():
    """Test stability for a series containing NaN values."""
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_stability(data)

def test_calculate_stability_max_lag_exceeds_length():
    """Test stability when max_lag exceeds the series length."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_stability(data, max_lag=10)
    assert np.isnan(result), f"Expected NaN when max_lag exceeds series length. Got: {result}"

def test_calculate_stability_zero_variance_series():
    """Test stability for a series with zero variance."""
    data = pd.Series([1, 1, 1, 1])
    result = calculate_stability(data)
    assert result == 1.0, f"Expected stability 1.0 for zero variance series. Got: {result}"

def test_calculate_stability_custom_max_lag():
    """Test stability with a custom max_lag."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_stability(data, max_lag=3)
    assert 0 <= result <= 1, f"Stability should be between 0 and 1. Got: {result}"
