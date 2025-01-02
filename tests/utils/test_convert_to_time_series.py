import pytest
import pandas as pd
import numpy as np
from interpreTS.utils.data_conversion import convert_to_time_series
from interpreTS.core.time_series_data import TimeSeriesData


def test_convert_to_time_series_with_dataframe():
    """Test conversion of a DataFrame to TimeSeriesData."""
    data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."


def test_convert_to_time_series_with_series():
    """Test conversion of a Series to TimeSeriesData."""
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."


def test_convert_to_time_series_with_ndarray():
    """Test conversion of a numpy array to TimeSeriesData."""
    data = np.array([1, 2, 3, 4, 5])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    expected_data = pd.Series(data)
    assert ts_data.data.equals(expected_data), "Data mismatch after conversion."


def test_convert_to_time_series_with_empty_dataframe():
    """Test conversion of an empty DataFrame to TimeSeriesData."""
    data = pd.DataFrame()
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."


def test_convert_to_time_series_with_empty_series():
    """Test conversion of an empty Series to TimeSeriesData."""
    data = pd.Series(dtype=float)
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."


def test_convert_to_time_series_with_empty_ndarray():
    """Test conversion of an empty numpy array to TimeSeriesData."""
    data = np.array([])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    expected_data = pd.Series(data)
    assert ts_data.data.equals(expected_data), "Data mismatch after conversion."


def test_convert_to_time_series_with_invalid_type():
    """Test conversion with an invalid data type."""
    data = {"value": [1, 2, 3, 4, 5]}  # Dictionary is not a valid type
    with pytest.raises(TypeError, match="Data must be of type pandas DataFrame, Series, or numpy array."):
        convert_to_time_series(data)


def test_convert_to_time_series_with_mixed_dataframe():
    """Test conversion of a DataFrame with mixed data types to TimeSeriesData."""
    data = pd.DataFrame({"value": [1, 2, 3, "a", 5]})
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."


def test_convert_to_time_series_with_large_ndarray():
    """Test conversion of a large numpy array to TimeSeriesData."""
    data = np.random.rand(10**6)
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    expected_data = pd.Series(data)
    assert ts_data.data.equals(expected_data), "Data mismatch after conversion."


def test_convert_to_time_series_with_nan_values():
    """Test conversion of data containing NaN values to TimeSeriesData."""
    data = pd.Series([1, np.nan, 3, 4, 5])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."


def test_convert_to_time_series_with_datetime_index():
    """Test conversion of a time series with a datetime index."""
    dates = pd.date_range("2023-01-01", periods=5)
    data = pd.Series([1, 2, 3, 4, 5], index=dates)
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."

def test_convert_to_time_series_with_datetime_index():
    """Test conversion of a time series with a datetime index."""
    dates = pd.date_range("2023-01-01", periods=5)
    data = pd.Series(["a", "b", "c", "d", "e"], index=dates)
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "Expected TimeSeriesData instance."
    assert ts_data.data.equals(data), "Data mismatch after conversion."