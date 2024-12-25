import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_below_1st_decile import calculate_below_1st_decile

# Test a basic case where some values are below the 1st decile
def test_below_1st_decile_basic_case():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_below_1st_decile(data, training_data)
    expected = 0.2  # Only 1 (out of 5 values) is below the 1st decile of training data
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when no values are below the 1st decile
def test_below_1st_decile_no_values_below_decile():
    data = pd.Series([5, 6, 7, 8, 9])
    training_data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_below_1st_decile(data, training_data)
    expected = 0.0  # No values in data are below the 1st decile of training data
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when all values are below the 1st decile
def test_below_1st_decile_all_values_below_decile():
    data = pd.Series([1, 1, 1, 1, 1])
    training_data = pd.Series([5, 6, 7, 8, 9, 10])
    result = calculate_below_1st_decile(data, training_data)
    expected = 1.0  # All values in data are below the 1st decile of training data
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when input data is a numpy array
def test_below_1st_decile_numpy_input():
    data = np.array([1, 2, 3, 4, 5])
    training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_below_1st_decile(data, training_data)
    expected = 0.2
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when data contains a single value
def test_below_1st_decile_single_value_data():
    data = pd.Series([1])
    training_data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_below_1st_decile(data, training_data)
    expected = 1.0  # Single value in data is below the 1st decile
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"
