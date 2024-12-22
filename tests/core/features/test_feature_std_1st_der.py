import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_std_1st_der import calculate_std_1st_der

# Test for normal input data with a consistent increase
def test_std_1st_der_basic_case():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_std_1st_der(data)
    expected = 0.0  # Consistent increase, constant first derivative
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for data with fluctuations
def test_std_1st_der_with_fluctuations():
    data = pd.Series([1, 3, 2, 4, 1])
    result = calculate_std_1st_der(data)
    expected = np.std(np.gradient([1, 3, 2, 4, 1]))
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for data containing a single value
def test_std_1st_der_single_value():
    data = pd.Series([10])
    result = calculate_std_1st_der(data)
    assert result == 0.0, f"Expected 0.0, got {result}"

# Test for data containing two values
def test_std_1st_der_two_values():
    data = pd.Series([10, 20])
    result = calculate_std_1st_der(data)
    expected = 0.0  # Uniform derivative change
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for data with negative values
def test_std_1st_der_negative_values():
    data = pd.Series([-5, -10, -15, -20])
    result = calculate_std_1st_der(data)
    expected = 0.0  # Constant decrease
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for numpy array input
def test_std_1st_der_numpy_array():
    data = np.array([1, 3, 2, 4, 1])
    result = calculate_std_1st_der(data)
    expected = np.std(np.gradient(data))
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test for a large dataset
def test_std_1st_der_large_data():
    data = pd.Series(np.linspace(0, 1000, 10000))
    result = calculate_std_1st_der(data)
    expected = 0.0  # Uniform increase
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"
