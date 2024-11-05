import pytest
import pandas as pd
import numpy as np
from interpreTS.utils.data_conversion import convert_to_time_series
from interpreTS.core.time_series_data import TimeSeriesData

def test_convert_pandas_series():
    """
    Test conversion of pandas Series to TimeSeriesData.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "The result should be an instance of TimeSeriesData"
    assert ts_data.data.equals(data), "The data should match the original pandas Series"

def test_convert_numpy_array():
    """
    Test conversion of numpy array to TimeSeriesData.
    """
    data = np.array([1, 2, 3, 4, 5])
    ts_data = convert_to_time_series(data)
    assert isinstance(ts_data, TimeSeriesData), "The result should be an instance of TimeSeriesData"
    assert ts_data.data.equals(pd.Series(data)), "The data should match the converted pandas Series"

def test_convert_invalid_type():
    """
    Test that convert_to_time_series raises an error with invalid data types.
    """
    with pytest.raises(TypeError):
        convert_to_time_series("invalid data")
