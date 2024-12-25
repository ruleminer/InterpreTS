import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_heterogeneity import calculate_heterogeneity

# Test heterogeneity for time series with positive mean and variability
def test_heterogeneity_positive_mean():
    data = pd.Series([1, 2, 3, 4, 10])
    result = calculate_heterogeneity(data)
    assert result > 0, "Heterogeneity should be positive for non-uniform data"

# Test heterogeneity for time series with no variability (constant values)
def test_heterogeneity_no_variability():
    data = pd.Series([5, 5, 5, 5])
    result = calculate_heterogeneity(data)
    assert result == 0, "Heterogeneity should be 0 for data with no variability"

# Test heterogeneity for time series that includes negative values
def test_heterogeneity_with_negative_values():
    data = pd.Series([-10, -5, 0, 5, 15])
    result = calculate_heterogeneity(data)
    assert result > 0, "Heterogeneity should be positive even with negative values"

# Test heterogeneity for time series that includes zero values
def test_heterogeneity_with_zeros():
    data = pd.Series([0, 1, 2, 3, 4])
    result = calculate_heterogeneity(data)
    assert result > 0, "Heterogeneity should handle zero values correctly"

# Test heterogeneity for time series with mean zero
def test_heterogeneity_mean_zero():
    data = pd.Series([-1, 1, -1, 1])
    result = calculate_heterogeneity(data)
    assert np.isnan(result), "Heterogeneity should be NaN for a series with zero mean"

# Test heterogeneity for numpy array input
def test_heterogeneity_numpy_array():
    data = np.array([1, 2, 3, 4, 10])
    result = calculate_heterogeneity(pd.Series(data))
    assert result > 0, "Heterogeneity should be positive for non-uniform data"

# Test heterogeneity for time series with very large values
def test_heterogeneity_large_values():
    data = pd.Series([1e9, 2e9, 3e9, 4e9, 5e9])
    result = calculate_heterogeneity(data)
    assert result > 0, "Heterogeneity should handle large values correctly"

# Test heterogeneity for time series with very small values
def test_heterogeneity_small_values():
    data = pd.Series([1e-9, 2e-9, 3e-9, 4e-9, 5e-9])
    result = calculate_heterogeneity(data)
    assert result > 0, "Heterogeneity should handle small values correctly"

# Test heterogeneity for time series with a single value
def test_heterogeneity_single_value():
    data = pd.Series([42])
    result = calculate_heterogeneity(data)
    assert result == 0, "Heterogeneity should be 0 for a single value (no variability)"

# Test heterogeneity for time series with only negative values
def test_heterogeneity_negative_only():
    data = pd.Series([-10, -20, -30, -40, -50])
    result = calculate_heterogeneity(data)
    assert result > 0, "Heterogeneity should be positive for negative-only values"
