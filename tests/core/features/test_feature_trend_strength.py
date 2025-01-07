import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_trend_strength import calculate_trend_strength

# Test for a perfect increasing trend
def test_trend_strength_increasing_data():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_trend_strength(data)
    expected = 1.0  # Perfect trend
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for a perfect decreasing trend
def test_trend_strength_decreasing_data():
    data = pd.Series([5, 4, 3, 2, 1])
    result = calculate_trend_strength(data)
    expected = 1.0  # Perfect trend
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for a series with no trend
def test_trend_strength_no_trend():
    data = pd.Series([1, 1, 1, 1, 1])
    result = calculate_trend_strength(data)
    expected = 0.0  # No trend
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for data with noise
def test_trend_strength_with_noise():
    data = pd.Series([1, 2, 3, 2, 4, 5])
    result = calculate_trend_strength(data)
    assert 0.0 < result < 1.0, f"Expected trend strength between 0 and 1, got {result}"

# Test for a single value in data
def test_trend_strength_single_value():
    data = pd.Series([10])
    result = calculate_trend_strength(data)
    expected = np.nan  # Insufficient data for a trend
    assert np.isnan(result), f"Expected NaN, got {result}"

# Test for exactly two values
def test_trend_strength_two_values():
    data = pd.Series([1, 2])
    result = calculate_trend_strength(data)
    expected = 1.0  # Perfect trend with two points
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for numpy array input
def test_trend_strength_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    result = calculate_trend_strength(data)
    expected = 1.0  # Perfect trend
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for random data with no apparent trend
def test_trend_strength_with_random_data():
    rng = np.random.default_rng(42)
    data = pd.Series(rng.normal(size=100))
    result = calculate_trend_strength(data)
    assert 0.0 <= result <= 1.0, f"Expected trend strength between 0 and 1, got {result}"
