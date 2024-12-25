import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from interpreTS.core.features.feature_linearity import calculate_linearity

# Test linearity for a perfectly linear series
def test_calculate_linearity_perfect_linear():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_linearity(data)
    assert result == pytest.approx(1.0, abs=0.001), f"Expected 1.0 for perfectly linear data. Got: {result}"

# Test linearity for a non-linear series
def test_calculate_linearity_non_linear():
    data = pd.Series([1, 1, 2, 2, 3, 5, 8, 13])  # Fibonacci series
    result = calculate_linearity(data)
    assert result < 1.0, f"Expected less than 1.0 for non-linear data. Got: {result}"

# Test linearity for constant series
def test_calculate_linearity_constant_series():
    data = pd.Series([5, 5, 5, 5])
    result = calculate_linearity(data)
    assert result == pytest.approx(0.0), f"Expected 0.0 for constant data. Got: {result}"

# Test linearity for a numpy array input
def test_calculate_linearity_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    result = calculate_linearity(data)
    assert result == pytest.approx(1.0, abs=0.001), f"Expected 1.0 for perfectly linear numpy array. Got: {result}"

# Test linearity with normalize=False
def test_calculate_linearity_no_normalization():
    data = pd.Series([10, 20, 30, 40, 50])  # Scaled version of a linear series
    result_with_norm = calculate_linearity(data, normalize=True)
    result_without_norm = calculate_linearity(data, normalize=False)
    assert result_with_norm == pytest.approx(result_without_norm, abs=0.001), (
        f"Normalization should not affect linearity for perfectly linear data. Got: {result_with_norm} vs {result_without_norm}"
    )

# Test linearity with first derivative (use_derivative=True)
def test_calculate_linearity_with_derivative():
    data = pd.Series([1, 3, 6, 10, 15])  # Non-linear data
    result_no_derivative = calculate_linearity(data, use_derivative=False)
    result_with_derivative = calculate_linearity(data, use_derivative=True)
    assert result_with_derivative > result_no_derivative, (
        f"Expected derivative to increase linearity for cumulative data. Got: {result_with_derivative} vs {result_no_derivative}"
    )
