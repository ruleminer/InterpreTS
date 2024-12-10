import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_heterogeneity import heterogeneity

def test_heterogeneity_positive_mean():
    """
    Test heterogeneity on a time series with a positive mean and variability.
    """
    data = pd.Series([1, 2, 3, 4, 10])
    result = heterogeneity(data)
    assert result > 0, "Heterogeneity should be positive for non-uniform data"

def test_heterogeneity_no_variability():
    """
    Test heterogeneity on a time series with no variability.
    """
    data = pd.Series([5, 5, 5, 5])
    result = heterogeneity(data)
    assert result == 0, "Heterogeneity should be 0 for data with no variability"

def test_heterogeneity_with_negative_values():
    """
    Test heterogeneity on a time series that includes negative values.
    """
    data = pd.Series([-10, -5, 0, 5, 15])
    result = heterogeneity(data)
    assert result > 0, "Heterogeneity should be positive even with negative values"

def test_heterogeneity_with_zeros():
    """
    Test heterogeneity on a time series that includes zero values.
    """
    data = pd.Series([0, 1, 2, 3, 4])
    result = heterogeneity(data)
    assert result > 0, "Heterogeneity should handle zero values correctly"

def test_heterogeneity_empty_series():
    """
    Test heterogeneity on an empty time series.
    """
    data = pd.Series([], dtype="float64")
    result = heterogeneity(data)
    assert np.isnan(result), "Heterogeneity should be NaN for an empty series"

def test_heterogeneity_with_nan_values():
    """
    Test heterogeneity on a time series containing NaN values.
    """
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        heterogeneity(data)

def test_heterogeneity_mean_zero():
    """
    Test heterogeneity on a time series with a mean of zero.
    """
    data = pd.Series([-1, 1, -1, 1])
    result = heterogeneity(data)
    assert np.isnan(result), "Heterogeneity should be NaN for a series with zero mean"

def test_heterogeneity_numpy_array():
    """
    Test heterogeneity on a numpy array (alternative data format).
    """
    data = np.array([1, 2, 3, 4, 10])
    result = heterogeneity(pd.Series(data))
    assert result > 0, "Heterogeneity should be positive for non-uniform data"

def test_heterogeneity_large_values():
    """
    Test heterogeneity on a time series with very large values.
    """
    data = pd.Series([1e9, 2e9, 3e9, 4e9, 5e9])
    result = heterogeneity(data)
    assert result > 0, "Heterogeneity should handle large values correctly"

def test_heterogeneity_small_values():
    """
    Test heterogeneity on a time series with very small values.
    """
    data = pd.Series([1e-9, 2e-9, 3e-9, 4e-9, 5e-9])
    result = heterogeneity(data)
    assert result > 0, "Heterogeneity should handle small values correctly"

def test_heterogeneity_single_value():
    """
    Test heterogeneity on a time series with a single value.
    """
    data = pd.Series([42])
    result = heterogeneity(data)
    assert result == 0, "Heterogeneity should be 0 for a single value (no variability)"

def test_heterogeneity_negative_only():
    """
    Test heterogeneity on a time series with only negative values.
    """
    data = pd.Series([-10, -20, -30, -40, -50])
    result = heterogeneity(data)
    assert result > 0, "Heterogeneity should be positive for negative-only values"
