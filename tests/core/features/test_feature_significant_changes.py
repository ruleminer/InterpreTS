import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_significant_changes import calculate_significant_changes

# Test for a standard case with varying values
def test_significant_changes_standard_case():
    data = pd.Series([1, 2, 1.5, 3, 2.5, 5, 4.5])
    result = calculate_significant_changes(data)
    expected = 0.0  # 0 significant changes out of 6 differences
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for a series with a single point
def test_significant_changes_single_point():
    data = pd.Series([1])
    result = calculate_significant_changes(data)
    expected = 0.0  # Cannot calculate differences
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for constant values
def test_significant_changes_constant_values():
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_significant_changes(data)
    expected = 0.0  # No differences, hence no significant changes
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for a series containing NaN values
def test_significant_changes_with_nan():
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_significant_changes(data)

# Test for a series with two points
def test_significant_changes_two_points():
    data = pd.Series([1, 2])
    result = calculate_significant_changes(data)
    expected = 0.0  # Only one difference, not enough for "significant changes"
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for negative values
def test_significant_changes_negative_values():
    data = pd.Series([-5, -10, -5, -15, -5, -20])
    result = calculate_significant_changes(data)
    assert result >= 0.0, f"Expected a positive proportion, got {result}"

# Test for an empty series
def test_significant_changes_empty_data():
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_significant_changes(data)

# Test for numpy array input
def test_significant_changes_numpy_array():
    data = np.array([1, 2, 1.5, 3, 2.5, 5, 4.5])
    result = calculate_significant_changes(data)
    expected = 0.0
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for monotonic increasing data
def test_significant_changes_monotonic_increasing():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_significant_changes(data)
    expected = 0.0  # No significant changes
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for monotonic decreasing data
def test_significant_changes_monotonic_decreasing():
    data = pd.Series([5, 4, 3, 2, 1])
    result = calculate_significant_changes(data)
    expected = 0.0  # No significant changes
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"
