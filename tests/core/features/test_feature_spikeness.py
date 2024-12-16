import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_spikeness import calculate_spikeness

def test_calculate_spikeness_simple_case():
    """Test spikeness for a simple series."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_spikeness(data)
    assert result == pytest.approx(0.0), f"Expected 0.0 for symmetric data. Got: {result}"

def test_calculate_spikeness_positive_skew():
    """Test spikeness for a positively skewed series."""
    data = pd.Series([1, 1, 2, 3, 8])
    result = calculate_spikeness(data)
    assert result > 0, f"Expected positive spikeness. Got: {result}"

def test_calculate_spikeness_negative_skew():
    """Test spikeness for a negatively skewed series."""
    data = pd.Series([10, 8, 7, 5, 2])  # Negative skew: long tail on the left
    print(f"Testing data: {data.tolist()}")
    result = calculate_spikeness(data)
    expected_result = data.skew()
    print(f"Expected spikeness: {expected_result}, Got: {result}")
    assert result == pytest.approx(expected_result, abs=0.001), (
        f"Unexpected spikeness. Expected: {expected_result}, Got: {result}"
    )
    assert result < 0, f"Expected negative spikeness. Got: {result}"


def test_calculate_spikeness_constant_data():
    """Test spikeness for a constant series."""
    data = pd.Series([5, 5, 5, 5])
    result = calculate_spikeness(data)
    assert result == pytest.approx(0.0), f"Expected 0.0 for constant data. Got: {result}"

def test_calculate_spikeness_with_nan():
    """Test spikeness for a series containing NaN values."""
    data = pd.Series([1, 2, np.nan, 4, 5])
    result = calculate_spikeness(data)
    expected_result = pd.Series([1, 2, 4, 5]).skew()  # Exclude NaN and calculate expected spikeness
    assert result == pytest.approx(expected_result, abs=0.001), f"Unexpected result for data with NaN. Got: {result}"

def test_calculate_spikeness_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_spikeness(data)

def test_calculate_spikeness_all_nan():
    """Test spikeness for a series with all NaN values."""
    data = pd.Series([np.nan, np.nan, np.nan])
    result = calculate_spikeness(data)
    assert np.isnan(result), f"Expected NaN for all-NaN series. Got: {result}"

def test_calculate_spikeness_numpy_array():
    """Test spikeness for a numpy array."""
    data = np.array([1, 2, 3, 4, 5])
    result = calculate_spikeness(data)
    assert result == pytest.approx(0.0), f"Expected 0.0 for symmetric numpy array. Got: {result}"

def test_calculate_spikeness_non_numeric():
    """Test spikeness for non-numeric data."""
    data = pd.Series(["a", "b", "c", "d"])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        calculate_spikeness(data)
