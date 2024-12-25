import pytest
import numpy as np
import pandas as pd
from interpreTS.core.features.feature_outliers_std import calculate_outliers_std

# Test when some values are outliers based on 3 standard deviations
def test_outliers_std_basic_case():
    data = pd.Series([1, 2, 3, 4, 100])  # 100 is an outlier
    training_data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_outliers_std(data, training_data)
    expected = 0.2  # 1 out of 5 values is an outlier
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when no values are outliers
def test_outliers_std_no_outliers():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_outliers_std(data, training_data)
    expected = 0.0  # No outliers
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when all values are outliers
def test_outliers_std_all_outliers():
    data = pd.Series([100, 200, 300])
    training_data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_outliers_std(data, training_data)
    expected = 1.0  # All values are outliers
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test with a single value in the training data
def test_outliers_std_single_training_point():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([5])
    result = calculate_outliers_std(data, training_data)
    expected = 0.8  # All values are outliers due to std_dev == 0
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test with numpy array inputs
def test_outliers_std_numpy_data():
    data = np.array([1, 2, 3, 100])
    training_data = np.array([1, 2, 3, 4, 5])
    result = calculate_outliers_std(data, training_data)
    expected = 0.25  # 1 out of 4 values is an outlier
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test with mixed input types (pandas Series and numpy array)
def test_outliers_std_mixed_input():
    data = pd.Series([1, 2, 3, 100])
    training_data = np.array([1, 2, 3, 4, 5])
    result = calculate_outliers_std(data, training_data)
    expected = 0.25  # 1 out of 4 values is an outlier
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"
