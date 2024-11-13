import pytest
import pandas as pd
import numpy as np

from data_validation import validate_time_series_data

def test_validate_pandas_series():
    """
    Test validation of a valid pandas Series.
    """
    data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5))
    assert validate_time_series_data(data) == True, "The validation should pass for a valid pandas Series"

def test_validate_with_nan_values():
    """
    Test that validation raises an error for data containing NaN values.
    """
    data = pd.Series([1, np.nan, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5))
    with pytest.raises(ValueError):
        validate_time_series_data(data)

def test_validate_without_datetime_index():
    """
    Test that validation raises an error for a Series without DateTime index.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        validate_time_series_data(data, require_datetime_index=True)

def test_validate_numpy_array():
    """
    Test validation of a valid numpy array.
    """
    data = np.array([1, 2, 3, 4, 5])
    assert validate_time_series_data(data) == True, "The validation should pass for a valid numpy array"

def test_validate_invalid_type():
    """
    Test that validation raises an error for an invalid data type.
    """
    data = "invalid data"
    with pytest.raises(TypeError):
        validate_time_series_data(data)
