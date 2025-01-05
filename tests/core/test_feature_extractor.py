import re
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from interpreTS.utils.feature_loader import Features
from interpreTS.core.feature_extractor import FeatureExtractor

@pytest.fixture
def mock_feature_extractor():
    feature_functions = {
        Features.MEAN: lambda data, **params: data.mean(),
        Features.VARIANCE: lambda data, **params: data.var()
    }
    mock_task_manager = MagicMock()
    mock_task_manager._validate_parameters = MagicMock()
    mock_task_manager._execute_dask = MagicMock()
    mock_task_manager._generate_tasks = MagicMock(return_value=[])
    mock_task_manager._execute_parallel = MagicMock()
    mock_task_manager._execute_sequential = MagicMock()
    mock_task_manager._process_window = MagicMock(return_value={'id': 1, 'value': 30})


    feature_extractor = FeatureExtractor(
        features=[Features.MEAN, Features.VARIANCE],
        feature_params={},
        window_size=5,
        stride=1,
        id_column="id",
        sort_column="time",
        feature_column="value",
        group_by="id",
    )

    feature_extractor.feature_functions = feature_functions
    feature_extractor.task_manager = mock_task_manager

    return feature_extractor

# Test initialization
def test_feature_extractor_initialization(mock_feature_extractor):
    assert mock_feature_extractor.features == [Features.MEAN, Features.VARIANCE]
    assert mock_feature_extractor.window_size == 5
    assert mock_feature_extractor.stride == 1

# Test window_size and stride with a valid time-based string
def test_feature_extractor_initialization_time_based(mock_feature_extractor):
    mock_feature_extractor.window_size = '1h'
    mock_feature_extractor.stride = '5min'

    assert mock_feature_extractor.window_size == '1h'
    assert mock_feature_extractor.stride == '5min'

# Test extract_features with empty data
def test_extract_features_empty_data(mock_feature_extractor):
    data = pd.DataFrame()
    result = mock_feature_extractor.extract_features(data)
    assert result.empty

# Test extract_features with sequential mode
def test_extract_features_sequential(mock_feature_extractor):
    data = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "value": [10, 20, 30, 40]
    })
    mock_feature_extractor.task_manager._execute_sequential.return_value = [
        {"mean_value": 15.0, "variance_value": 25.0},
        {"mean_value": 35.0, "variance_value": 25.0}
    ]
    result = mock_feature_extractor.extract_features(data, mode="sequential")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

def test_validate_time_based_window_and_stride(mock_feature_extractor):
    # Test with missing datetime index
    data = pd.DataFrame({
        'id': [1, 1, 2, 2],
        'time': [1, 2, 1, 2],
        'value': [10, 20, 30, 40]
    })
    data.set_index('time', inplace=True)

    mock_feature_extractor.window_size = '1h'
    mock_feature_extractor.stride = '5min'

    grouped_data = data.groupby('id')

    with pytest.raises(ValueError, match=re.escape("Time-based window_size and stride require a time-indexed DataFrame with regular frequency.")):
        mock_feature_extractor.validate_data_frequency(grouped_data)

    # Test with datetime index but no frequency
    data.index = pd.to_datetime(data.index)
    data.index.freq = None

    grouped_data = data.groupby('id')

    with pytest.raises(ValueError, match=re.escape("Data index does not have a defined frequency. Use `.resample()` to align your data.")):
        mock_feature_extractor.validate_data_frequency(grouped_data)


# Test group_data
def test_group_data(mock_feature_extractor):
    data = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "value": [10, 20, 30, 40]
    })
    groups = list(mock_feature_extractor.group_data(data))
    assert len(groups) == 2

# Test extract_features_stream
def test_extract_features_stream(mock_feature_extractor):
    data_stream = [
        {"id": 1, "value": 10},
        {"id": 1, "value": 20},
        {"id": 2, "value": 30},
        {"id": 2, "value": 40},
    ]
    results = list(mock_feature_extractor.extract_features_stream(data_stream))
    assert len(results) == 0  # Mocked results do not yield any real features

# Test add_custom_feature
def test_add_custom_feature(mock_feature_extractor):
    def custom_feature(data, **params):
        return data.max() - data.min()

    mock_feature_extractor.add_custom_feature(
        name="range",
        function=custom_feature,
        metadata={"level": "easy", "description": "Range of values."}
    )

    assert "range" in mock_feature_extractor.feature_functions
    assert mock_feature_extractor.feature_metadata["range"]["description"] == "Range of values."

# Test invalid custom feature addition
def test_add_invalid_custom_feature(mock_feature_extractor):
    with pytest.raises(ValueError):
        mock_feature_extractor.add_custom_feature(name="mean", function=lambda x: x)

    with pytest.raises(ValueError):
        mock_feature_extractor.add_custom_feature(name="custom", function=None)

# Test group_features_by_interpretability
def test_group_features_by_interpretability(mock_feature_extractor):
    groups = mock_feature_extractor.group_features_by_interpretability()
    assert "easy" in groups
    assert isinstance(groups["easy"], list)

def test_extract_features_with_time_based_window(mock_feature_extractor):
    data = pd.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "time": pd.to_datetime([
            '2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00',
            '2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00'
            ]),
        "value": [10, 20, 30, 40, 50, 60]
    })

    data.set_index('time', inplace=True)

    mock_feature_extractor.window_size = '1h'
    mock_feature_extractor.stride = '1h'

    mock_feature_extractor.task_manager._execute_sequential.return_value = [
        {"mean_value": 15.0, "variance_value": 25.0},
        {"mean_value": 35.0, "variance_value": 25.0}
    ]
    result = mock_feature_extractor.extract_features(data, mode="sequential")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

def test_extract_features_stream_time_based(mock_feature_extractor):
    data_stream = [
        {"id": 1, "time": '2023-01-01 00:00:00', "value": 10},
        {"id": 1, "time": '2023-01-01 01:00:00', "value": 20},
        {"id": 2, "time": '2023-01-01 00:00:00', "value": 30},
        {"id": 2, "time": '2023-01-01 01:00:00', "value": 40},
    ]
    mock_feature_extractor.window_size = '1h'
    mock_feature_extractor.stride = '1h'
    mock_feature_extractor.id_column = 'id'
    mock_feature_extractor.sort_column = 'time'

    results = list(mock_feature_extractor.extract_features_stream(data_stream))


    print(results)
    assert len(results) > 0
    assert 'id' in results[0]
    assert 'value' in results[0]

