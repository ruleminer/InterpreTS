import pytest
import pandas as pd
from interpreTS.core.time_series_data import TimeSeriesData

def test_time_series_initialization():
    """
    Test the initialization of TimeSeriesData with a pandas Series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = TimeSeriesData(data)
    assert ts_data.data.equals(data), "The data should be initialized correctly"

def test_time_series_resample():
    """
    Test the resampling of TimeSeriesData with a datetime index.
    """
    data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5, freq="D"))
    ts_data = TimeSeriesData(data)
    resampled_data = ts_data.resample("2D")
    assert len(resampled_data.data) == 3, "The resampled data should have 3 points for 2-day frequency"

def test_time_series_split():
    """
    Test the split functionality of TimeSeriesData.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = TimeSeriesData(data)
    train_data, test_data = ts_data.split(0.6)
    assert len(train_data.data) == 3, "The training set should have 60% of the data"
    assert len(test_data.data) == 2, "The test set should have 40% of the data"
