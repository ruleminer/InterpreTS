import pytest
import pandas as pd
import numpy as np
import dask.dataframe as dd
from unittest.mock import MagicMock, patch
from interpreTS.utils.data_validation import validate_time_series_data
from interpreTS.utils.feature_loader import FeatureLoader
from interpreTS.utils.task_manager import TaskManager

# Fixture for initializing TaskManager with mock feature functions and validation requirements
@pytest.fixture
def task_manager():
    feature_functions = {
        "mock_feature": lambda x: x.sum(),
    }
    validation_requirements = {
        "mock_feature": {
            "require_datetime_index": False,
            "allow_nan": False,
        }
    }
    return TaskManager(
        feature_functions=feature_functions,
        window_size=3,
        features=["mock_feature"],
        stride=1,
        feature_params={},
        validation_requirements=validation_requirements,
    )

# Fixture for mocking FeatureLoader to return a predefined list of available features
@pytest.fixture
def mock_feature_loader():
    with patch("interpreTS.utils.feature_loader.FeatureLoader.available_features", return_value=["mock_feature"]):
        yield

# Test calculation of a valid feature
def test_calculate_feature(task_manager):
    data = pd.Series([1, 2, 3])
    result = task_manager._calculate_feature("mock_feature", data, {})
    assert result == 6

# Test attempting to calculate an unsupported feature
def test_calculate_feature_invalid(task_manager):
    with pytest.raises(ValueError, match="Feature 'invalid_feature' is not supported."):
        task_manager._calculate_feature("invalid_feature", pd.Series([1, 2, 3]), {})

# Test validation of valid input parameters for TaskManager
def test_validate_parameters_valid(mock_feature_loader):
    TaskManager._validate_parameters(
        features=["mock_feature"],
        feature_params={},
        window_size=5,
        stride=1,
        id_column=None,
        sort_column=None,
    )

# Test validation of invalid feature names in input parameters
def test_validate_parameters_invalid_feature(mock_feature_loader):
    with pytest.raises(ValueError, match=r"The following features are invalid or not implemented: \['invalid_feature'\]."):
        TaskManager._validate_parameters(
            features=["invalid_feature"],
            feature_params={},
            window_size=5,
            stride=1,
            id_column=None,
            sort_column=None,
        )

# Test execution of feature extraction using Dask
def test_execute_dask(task_manager):
    pandas_data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    dask_data = dd.from_pandas(pandas_data, npartitions=1)
    grouped_data = [(None, pandas_data)]
    feature_columns = ["value"]

    with patch("dask.dataframe.DataFrame.map_partitions", return_value=dask_data):
        with patch("dask.dataframe.concat", return_value=dask_data):
            result = task_manager._execute_dask(grouped_data, feature_columns, progress_callback=None)

    assert result.equals(pandas_data)

# Test processing a single partition for feature extraction
def test_process_partition(task_manager):
    data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    result = task_manager._process_partition(data, ["value"], 3, 1)
    assert isinstance(result, pd.DataFrame)

# Test generation of tasks for feature extraction based on input data
def test_generate_tasks(task_manager):
    grouped_data = [(None, pd.DataFrame({"value": [1, 2, 3, 4, 5]}))]
    feature_columns = ["value"]
    tasks = task_manager._generate_tasks(grouped_data, feature_columns)
    assert len(tasks) == 3  # 3 overlapping windows of size 3

# Test parallel execution of feature extraction tasks
def test_execute_parallel(task_manager):
    tasks = [
        (pd.DataFrame({"value": [1, 2, 3]}), ["value"]),
        (pd.DataFrame({"value": [2, 3, 4]}), ["value"]),
    ]

    with patch("joblib.Parallel", return_value=[{"mock_feature_value": 6}, {"mock_feature_value": 9}]):
        result = task_manager._execute_parallel(tasks, n_jobs=-1, progress_callback=None, total_steps=len(tasks))
        assert isinstance(result, list)
        assert len(result) == len(tasks)

# Test sequential execution of feature extraction tasks
def test_execute_sequential(task_manager):
    tasks = [
        (pd.DataFrame({"value": [1, 2, 3]}), ["value"]),
        (pd.DataFrame({"value": [2, 3, 4]}), ["value"]),
    ]

    result = task_manager._execute_sequential(tasks, progress_callback=None, total_steps=len(tasks))
    assert isinstance(result, list)
    assert len(result) == len(tasks)

# Test processing a single window of data for feature extraction
def test_process_window(task_manager):
    window = pd.DataFrame({"value": [1, 2, 3]})
    feature_columns = ["value"]
    result = task_manager._process_window(window, feature_columns)
    assert "mock_feature_value" in result
    assert result["mock_feature_value"] == 6

# Test processing a single window with progress callback
def test_process_window_with_progress(task_manager):
    task = (pd.DataFrame({"value": [1, 2, 3]}), ["value"])

    def mock_progress_callback():
        pass

    with patch.object(task_manager, "_process_window", return_value={"mock_feature_value": 6}):
        result = task_manager._process_window_with_progress(task, mock_progress_callback)
        assert result == {"mock_feature_value": 6}

# Test validation of valid feature data
def test_validate_feature_data(task_manager):
    data = pd.Series([1, 2, 3])
    task_manager._validate_feature_data("mock_feature", data)  # Should pass without exceptions

# Test validation of invalid feature data with NaN values
def test_validate_feature_data_invalid(task_manager):
    data = pd.Series([np.nan, np.nan])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        task_manager._validate_feature_data("mock_feature", data)
