import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_outliers_iqr import calculate_outliers_iqr

# Test when there are no outliers in the data
def test_outliers_iqr_no_outliers():
    data = pd.Series([5, 6, 7, 8, 9])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_outliers_iqr(data, training_data)
    expected = 0.0
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when there are some outliers in the data
def test_outliers_iqr_some_outliers():
    data = pd.Series([1, 2, 100, 200, 300])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_outliers_iqr(data, training_data)
    expected = 0.6  # 3 out of 5 values are outliers
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when all data points are outliers
def test_outliers_iqr_all_outliers():
    data = pd.Series([100, 200, 300, 400, 500])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_outliers_iqr(data, training_data)
    expected = 1.0
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when the input data is empty
def test_outliers_iqr_empty_data():
    data = pd.Series([], dtype=float)
    training_data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_outliers_iqr(data, training_data)

# Test when the training data is empty
def test_outliers_iqr_empty_training_data():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_outliers_iqr(data, training_data)

# Test when data or training_data contain NaN values
def test_outliers_iqr_with_nan():
    data = pd.Series([1, 2, np.nan, 4, 5])
    training_data = pd.Series([1, 2, 3, 4, np.nan])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_outliers_iqr(data, training_data)

# Test with mixed data types
def test_outliers_iqr_mixed_types():
    data = pd.Series([1, 2, "three", 4, 5])
    training_data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        calculate_outliers_iqr(data, training_data)

# Test when inputs are numpy arrays
def test_outliers_iqr_numpy_input():
    data = np.array([5, 6, 100, 200, 9])
    training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_outliers_iqr(data, training_data)
    expected = 0.4  # 2 out of 5 values are outliers
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when data contains a single value
def test_outliers_iqr_single_data_point():
    data = pd.Series([10])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_outliers_iqr(data, training_data)
    expected = 0.0  # Single data point cannot be an outlier
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test when training data contains a single value
def test_outliers_iqr_single_training_point():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([5])
    result = calculate_outliers_iqr(data, training_data)
    expected = 0.6
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"
