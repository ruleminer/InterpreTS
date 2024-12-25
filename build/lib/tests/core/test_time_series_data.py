import pytest
import pandas as pd
from interpreTS.core.time_series_data import TimeSeriesData

def test_initialize_with_series():
    """
    Test initialization with a valid pandas Series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = TimeSeriesData(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_series_equal(ts_data.data, data)


def test_initialize_with_dataframe():
    """
    Test initialization with a valid pandas DataFrame.
    """
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    ts_data = TimeSeriesData(data)
    assert isinstance(ts_data, TimeSeriesData)
    pd.testing.assert_frame_equal(ts_data.data, data)


def test_initialize_with_invalid_data():
    """
    Test initialization with invalid data types.
    """
    data = [1, 2, 3]  # List is not allowed
    with pytest.raises(ValueError, match="Data must be a pandas Series or DataFrame."):
        TimeSeriesData(data)


def test_resample_with_datetime_index():
    """
    Test resampling with a valid DateTime index.
    """
    data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5, freq="D"))
    ts_data = TimeSeriesData(data)
    resampled_data = ts_data.resample("2D")
    assert isinstance(resampled_data, TimeSeriesData)
    expected_index = pd.date_range("2023-01-01", periods=3, freq="2D")
    pd.testing.assert_series_equal(resampled_data.data, data.resample("2D").mean())


def test_resample_without_datetime_index():
    """
    Test resampling without a DateTime index.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = TimeSeriesData(data)
    with pytest.raises(ValueError, match="Data must have a DateTime index for resampling."):
        ts_data.resample("2D")


def test_split_default_train_size():
    """
    Test the default train-test split (70% train, 30% test).
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = TimeSeriesData(data)
    train, test = ts_data.split()
    assert isinstance(train, TimeSeriesData)
    assert isinstance(test, TimeSeriesData)
    pd.testing.assert_series_equal(train.data, data[:3])
    pd.testing.assert_series_equal(test.data, data[3:])


def test_split_custom_train_size():
    """
    Test train-test split with a custom train size.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = TimeSeriesData(data)
    train, test = ts_data.split(train_size=0.4)
    assert isinstance(train, TimeSeriesData)
    assert isinstance(test, TimeSeriesData)
    pd.testing.assert_series_equal(train.data, data[:2])
    pd.testing.assert_series_equal(test.data, data[2:])


def test_split_edge_cases():
    """
    Test train-test split edge cases (e.g., train size 0 or 1).
    """
    data = pd.Series([1, 2, 3, 4, 5])
    ts_data = TimeSeriesData(data)

    # Train size = 0
    train, test = ts_data.split(train_size=0)
    assert train.data.empty
    pd.testing.assert_series_equal(test.data, data)

    # Train size = 1
    train, test = ts_data.split(train_size=1)
    pd.testing.assert_series_equal(train.data, data)
    assert test.data.empty
