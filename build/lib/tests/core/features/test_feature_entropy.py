import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_entropy import calculate_entropy

def test_calculate_entropy_full_series():
    data = pd.Series([1, 3, 5, 2, 7])
    result = calculate_entropy(data)
    assert not np.isnan(result), "Entropy calculation failed for a non-constant series"

def test_calculate_entropy_with_identical_values():
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_entropy(data)
    assert result == 0.0, "Entropy should be 0 for constant data"

def test_calculate_entropy_with_low_variance():
    data = pd.Series([1, 1, 1, 2, 2, 2])
    result = calculate_entropy(data)
    assert result < 1, "Entropy should be low for data with low variance"

def test_calculate_entropy_with_high_variance():
    data = pd.Series([1, 3, 5, 7, 9])
    result = calculate_entropy(data)
    assert not np.isnan(result), "Entropy calculation failed for a high-variance data"

def test_calculate_entropy_empty_series():
    data = pd.Series([])
    result = calculate_entropy(data)
    assert np.isnan(result), "Entropy should be NaN for an empty series"

def test_calculate_entropy_single_value():
    data = pd.Series([42])
    result = calculate_entropy(data)
    assert result == 0.0, "Entropy should be 0 for a single-value data"

def test_calculate_entropy_with_non_numerical_data():
    data = pd.Series(['a', 'b', 'c', 'a'])
    try:
        result = calculate_entropy(data)
    except Exception as e:
        assert isinstance(e, TypeError), f"Expected TypeError, got {type(e)}"

def test_calculate_entropy_with_large_data():
    data = pd.Series(np.random.rand(100000))  # Large data with random values
    result = calculate_entropy(data)
    assert not np.isnan(result), "Entropy calculation failed for a large data"

def test_calculate_entropy_with_custom_range():
    data = pd.Series([1, 3, 5, 7, 9])
    result = calculate_entropy(data)
    assert result > 0, "Entropy should be greater than 0 for a non-constant data"

def test_calculate_entropy_with_large_variation():
    data = pd.Series(np.random.rand(10000))
    result = calculate_entropy(data)
    assert result > 0, "Entropy should be greater than 0 for random data"
