import pytest
import numpy as np
import pandas as pd
from interpreTS.utils.data_validation import validate_time_series_data

def test_validate_numeric_series():
    data = pd.Series([1, 2, 3, 4])
    assert validate_time_series_data(data, allow_nan=False)

def test_validate_numeric_array():
    data = np.array([1, 2, 3, 4])
    assert validate_time_series_data(data, allow_nan=False)

def test_validate_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="Input data is empty."):
        validate_time_series_data(data)

def test_validate_empty_array():
    data = np.array([])
    with pytest.raises(ValueError, match="Input data is empty."):
        validate_time_series_data(data)

def test_validate_nan_series():
    data = pd.Series([1, np.nan, 3])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        validate_time_series_data(data, allow_nan=False)

def test_validate_allow_nan():
    data = pd.Series([1, np.nan, 3])
    assert validate_time_series_data(data, allow_nan=True)

def test_validate_non_numeric_series():
    data = pd.Series(["a", "b", "c"])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        validate_time_series_data(data)

def test_validate_non_numeric_array():
    data = np.array(["a", "b", "c"])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        validate_time_series_data(data)

def test_validate_datetime_index():
    data = pd.Series([1, 2, 3, 4], index=pd.date_range("2023-01-01", periods=4))
    assert validate_time_series_data(data, require_datetime_index=True)

def test_validate_missing_datetime_index():
    data = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="Data must have a DateTime index for time-based operations."):
        validate_time_series_data(data, require_datetime_index=True)

def test_validate_min_length():
    data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Data must have at least 5 points."):
        validate_time_series_data(data, min_length=5)

def test_validate_min_length_dynamic():
    data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Data must have at least 4 points."):
        validate_time_series_data(data, min_length="len(data) + 1")

def test_validate_check_one_dimensional_series():
    data = pd.Series([1, 2, 3, 4])
    assert validate_time_series_data(data, check_one_dimensional=True)

def test_validate_check_one_dimensional_array():
    data = np.array([1, 2, 3, 4])
    assert validate_time_series_data(data, check_one_dimensional=True)

def test_validate_check_one_dimensional_invalid():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Data must be one-dimensional."):
        validate_time_series_data(data, check_one_dimensional=True)

def test_validate_nonzero_mean():
    data = pd.Series([1, -1, 1, -1])
    with pytest.raises(ValueError, match="Data mean must not be zero."):
        validate_time_series_data(data, check_nonzero_mean=True)

def test_validate_positive_parameters():
    data = pd.Series([1, 2, 3, 4])
    with pytest.raises(ValueError, match="Period must be a positive integer."):
        validate_time_series_data(data, validate_positive_parameters={"period": "Period must be a positive integer."}, period=-1)

def test_validate_positive_integer_params():
    data = pd.Series([1, 2, 3, 4])
    with pytest.raises(ValueError, match="window_size must be a positive integer."):
        validate_time_series_data(data, positive_integer_params=["window_size"], window_size=0)

def test_validate_dynamic_min_length():
    data = pd.Series([1, 2, 3, 4])
    assert validate_time_series_data(data, min_length=4)

def test_validate_dynamic_min_length_failure():
    data = pd.Series([1, 2])
    with pytest.raises(ValueError, match="Data must have at least 3 points."):
        validate_time_series_data(data, min_length="2 + 1")
