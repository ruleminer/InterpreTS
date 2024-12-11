import pytest
import pandas as pd
import numpy as np
from interpreTS.utils.data_conversion import convert_to_time_series
from interpreTS.core.time_series_data import TimeSeriesData

def test_convert_empty_series():
    """
    Test conversion of an empty pandas Series.
    """
    data = pd.Series(dtype=float)
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    assert ts_data.data.empty, "Expected an empty TimeSeriesData object."


def test_convert_empty_dataframe():
    """
    Test conversion of an empty pandas DataFrame.
    """
    data = pd.DataFrame()
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    assert ts_data.data.empty, "Expected an empty TimeSeriesData object."


def test_convert_empty_ndarray():
    """
    Test conversion of an empty numpy array.
    """
    data = np.array([])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    assert ts_data.data.empty, "Expected an empty TimeSeriesData object."


def test_convert_1d_numpy_array():
    """
    Test conversion of a 1D numpy array.
    """
    data = np.array([1, 2, 3, 4, 5])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_series_equal(ts_data.data, pd.Series(data))


def test_convert_2d_numpy_array():
    """
    Test conversion of a 2D numpy array.
    """
    data = np.array([[1, 2], [3, 4]])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_frame_equal(ts_data.data, pd.DataFrame(data))


def test_convert_invalid_ndarray():
    """
    Test conversion of a numpy array with more than 2 dimensions.
    """
    data = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError, match="Input numpy array must be 1D or 2D."):
        convert_to_time_series(data)


def test_convert_pandas_series():
    """
    Test conversion of a pandas Series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_series_equal(ts_data.data, data)


def test_convert_pandas_dataframe():
    """
    Test conversion of a pandas DataFrame.
    """
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_frame_equal(ts_data.data, data)


def test_convert_invalid_type():
    """
    Test conversion of an unsupported data type.
    """
    data = "invalid data"
    with pytest.raises(TypeError, match="Data must be of type pandas DataFrame, Series, or numpy array."):
        convert_to_time_series(data)


def test_convert_mixed_type_1d_numpy_array():
    """
    Test conversion of a 1D numpy array with mixed types.
    """
    data = np.array([1, "two", 3.0])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_series_equal(ts_data.data, pd.Series(data))


def test_convert_mixed_type_2d_numpy_array():
    """
    Test conversion of a 2D numpy array with mixed types.
    """
    data = np.array([[1, "two"], [3.0, 4]])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_frame_equal(ts_data.data, pd.DataFrame(data))
