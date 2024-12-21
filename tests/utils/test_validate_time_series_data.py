import pytest
import pandas as pd
import numpy as np
from interpreTS.utils.data_validation import validate_time_series_data

def test_validate_numeric_series():
    """Test validation of a numeric pandas Series."""
    data = pd.Series([1, 2, 3, 4, 5])
    assert validate_time_series_data(data) is True

def test_validate_series_with_nan_allowed():
    """Test validation of a Series with NaN values (allowed)."""
    data = pd.Series([1, np.nan, 3])
    assert validate_time_series_data(data) is True

def test_validate_series_with_nan_disallowed():
    """Test validation of a Series with NaN values (disallowed)."""
    data = pd.Series([1, np.nan, 3])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        validate_time_series_data(data, allow_nan=False)

def test_validate_non_numeric_series():
    """Test validation of a non-numeric pandas Series."""
    data = pd.Series(['a', 'b', 'c'])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        validate_time_series_data(data)

def test_validate_dataframe():
    """Test validation of a numeric pandas DataFrame."""
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert validate_time_series_data(data) is True

def test_validate_non_numeric_dataframe():
    """Test validation of a non-numeric pandas DataFrame."""
    data = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3]})
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        validate_time_series_data(data)

def test_validate_numpy_array():
    """Test validation of a numeric numpy array."""
    data = np.array([1, 2, 3, 4, 5])
    assert validate_time_series_data(data) is True

def test_validate_non_numeric_numpy_array():
    """Test validation of a non-numeric numpy array."""
    data = np.array(['a', 'b', 'c'])
    with pytest.raises(TypeError, match="Data must contain only numeric values."):
        validate_time_series_data(data)

def test_validate_empty_series():
    """Test validation of an empty pandas Series."""
    data = pd.Series([], dtype=float)
    assert validate_time_series_data(data) is True

def test_validate_empty_dataframe():
    """Test validation of an empty pandas DataFrame."""
    data = pd.DataFrame()
    assert validate_time_series_data(data) is True

def test_validate_empty_numpy_array():
    """Test validation of an empty numpy array."""
    data = np.array([])
    assert validate_time_series_data(data) is True

def test_validate_datetime_index_required():
    """Test validation when a DateTime index is required."""
    data = pd.Series([1, 2, 3], index=pd.date_range("2023-01-01", periods=3))
    assert validate_time_series_data(data, require_datetime_index=True) is True

def test_validate_no_datetime_index():
    """Test validation failure for missing DateTime index when required."""
    data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Data must have a DateTime index for time-based operations."):
        validate_time_series_data(data, require_datetime_index=True)

def test_validate_empty_series():
    """Test validation of an empty pandas Series."""
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        validate_time_series_data(data)

def test_validate_empty_dataframe():
    """Test validation of an empty pandas DataFrame."""
    data = pd.DataFrame()
    with pytest.raises(ValueError, match="Input data is empty."):
        validate_time_series_data(data)

def test_validate_empty_numpy_array():
    """Test validation of an empty numpy array."""
    data = np.array([])
    with pytest.raises(ValueError, match="Input data is empty."):
        validate_time_series_data(data)