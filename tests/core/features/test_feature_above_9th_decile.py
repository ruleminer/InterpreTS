import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_above_9th_decile import calculate_above_9th_decile

# Test a basic case with clear values above the 9th decile
def test_above_9th_decile_basic_case():
    data = pd.Series([8, 9, 10, 11, 12])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 0.6, f"Expected 0.6, got {result}"

# Test when no values are above the 9th decile
def test_above_9th_decile_no_values_above():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 0.0, f"Expected 0.0, got {result}"

# Test when all values are above the 9th decile
def test_above_9th_decile_all_values_above():
    data = pd.Series([11, 12, 13, 14, 15])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 1.0, f"Expected 1.0, got {result}"

# Test when the data series is empty
def test_above_9th_decile_empty_data():
    data = pd.Series([], dtype=float)
    training_data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_above_9th_decile(data, training_data)

# Test when the training data series is empty
def test_above_9th_decile_empty_training_data():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_above_9th_decile(data, training_data)

# Test when the data contains NaN values
def test_above_9th_decile_with_nan_in_data():
    data = pd.Series([1, 2, np.nan, 4, 5])
    training_data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Data and training data must not contain NaN values."):
        calculate_above_9th_decile(data, training_data)

# Test when the training data contains NaN values
def test_above_9th_decile_with_nan_in_training_data():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data and training data must not contain NaN values."):
        calculate_above_9th_decile(data, training_data)

# Test when inputs are numpy arrays
def test_above_9th_decile_numpy_input():
    data = np.array([8, 9, 10, 11, 12])
    training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 0.6, f"Expected 0.6, got {result}"

# Test when all values in data and training data are identical
def test_above_9th_decile_identical_values():
    data = pd.Series([5, 5, 5, 5, 5])
    training_data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 0.0, f"Expected 0.0, got {result}"

# Test behavior when some values are exactly at the 9th decile boundary
def test_above_9th_decile_edge_case_decile_boundary():
    data = pd.Series([8, 9, 10, 11, 12])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 0.6, f"Expected 0.6, got {result}"
