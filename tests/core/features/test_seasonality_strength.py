import pandas as pd
import numpy as np
import pytest
from interpreTS.core.features.seasonality_strength import calculate_seasonality_strength

# Test seasonality strength for periodic data
def test_seasonality_strength_valid_periodic():
    data = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3])
    result = calculate_seasonality_strength(data, period=3)
    assert result > 0.5, f"Expected strong seasonality, got {result}"

# Test seasonality strength for constant data
def test_seasonality_strength_constant_data():
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_seasonality_strength(data, period=2)
    assert result == 0.0, f"Expected 0 for constant data, got {result}"

# Test seasonality strength for insufficient data
def test_seasonality_strength_insufficient_data():
    data = pd.Series([1, 2])  # Less than the period
    result = calculate_seasonality_strength(data, period=3)
    assert np.isnan(result), f"Expected NaN for insufficient data, got {result}"

# Test seasonality strength for data containing NaN values
def test_seasonality_strength_with_nan():
    data = pd.Series([1, np.nan, 2, 3, 1, 2, 3])
    result = calculate_seasonality_strength(data, period=3)
    assert result >= 0.5, f"Expected strong seasonality ignoring NaN, got {result}"

# Test seasonality strength for random data
def test_seasonality_strength_random_data():
    np.random.seed(42)
    data = pd.Series(np.random.rand(100))
    result = calculate_seasonality_strength(data, period=12)
    assert result < 0.1, f"Expected low seasonality for random data, got {result}"

# Test seasonality strength for a single data point
def test_seasonality_strength_single_point():
    data = pd.Series([1])
    result = calculate_seasonality_strength(data, period=1)
    assert np.isnan(result), f"Expected NaN for a single data point, got {result}"

# Test seasonality strength for a large dataset with periodicity
def test_seasonality_strength_large_periodic_data():
    data = pd.Series(np.tile([1, 2, 3], 100))  # 300 points, periodicity of 3
    result = calculate_seasonality_strength(data, period=3)
    assert result > 0.9, f"Expected strong seasonality, got {result}"

# Test seasonality strength for periodic data with added noise
def test_seasonality_strength_periodic_with_noise():
    np.random.seed(42)
    periodic_data = np.tile([1, 2, 3], 10)  # Clear periodicity
    noise = np.random.normal(0, 0.1, len(periodic_data))
    data = pd.Series(periodic_data + noise)
    result = calculate_seasonality_strength(data, period=3)
    assert result > 0.5, f"Expected moderate to strong seasonality, got {result}"

# Test seasonality strength with invalid period
def test_seasonality_strength_invalid_period():
    data = pd.Series([1, 2, 3, 1, 2, 3])
    with pytest.raises(ValueError, match="Period must be a positive integer"):
        calculate_seasonality_strength(data, period=0)
