import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import dask.dataframe as dd
from joblib import Parallel, delayed
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from ..utils.data_validation import validate_time_series_data
from ..utils.feature_loader import FeatureLoader

class TaskManager:
    def __init__(self, feature_functions, window_size, features, stride, feature_params, validation_requirements):
        self.feature_functions = feature_functions
        self.window_size = window_size
        self.features = features
        self.stride = stride
        self.feature_params = feature_params
        self.validation_requirements = validation_requirements
        self.warning_registry = set()
    
    def _calculate_feature(self, feature_name, feature_data, params):
        """
        Calculate a specific feature.
        """
        if feature_name in self.feature_functions:
            return self.feature_functions[feature_name](feature_data, **params)
        else:
            raise ValueError(f"Feature '{feature_name}' is not supported.")
    
    @staticmethod
    def _validate_parameters(features, feature_params, window_size, stride, id_column, sort_column):
        """
        Validate input parameters to ensure they are valid.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        available_features = FeatureLoader.available_features()

        # Validate features
        if features is not None:
            if not isinstance(features, list):
                raise ValueError("Features must be a list or None.")
            invalid_features = [feature for feature in features if feature not in available_features]
            if invalid_features:
                raise ValueError(
                    f"The following features are invalid or not implemented: {invalid_features}. "
                    f"Available features are: {available_features}."
                )

        # Validate feature_params
        if feature_params is not None and not isinstance(feature_params, dict):
            raise ValueError("Feature parameters must be a dictionary or None.")

        # Validate window_size
        TaskManager._validate_time_or_numeric_parameter(
            window_size, "window_size", allow_nan=True
        )

        # Validate stride
        TaskManager._validate_time_or_numeric_parameter(
            stride, "stride", allow_nan=False
        )

        # Validate id_column
        if id_column is not None and not isinstance(id_column, str):
            raise ValueError("ID column must be a string or None.")

        # Validate sort_column
        if sort_column is not None and not isinstance(sort_column, str):
            raise ValueError("Sort column must be a string or None.")

    @staticmethod
    def _validate_time_or_numeric_parameter(param, param_name, allow_nan):
        """
        Validate a parameter that can be either a numeric value or a time-based string.

        Parameters
        ----------
        param : int, float, str, or None
            The parameter to validate.
        param_name : str
            Name of the parameter (for error messages).
        allow_nan : bool
            Whether to allow NaN as a valid value.

        Raises
        ------
        ValueError
            If the parameter is invalid.
        """
        from pandas.tseries.frequencies import to_offset

        if isinstance(param, (int, float)):
            if param <= 0:
                raise ValueError(f"{param_name} must be a positive number.")
        elif isinstance(param, str):
            try:
                offset = to_offset(param)
                if not hasattr(offset, "nanos"):
                    raise ValueError(
                        f"Unsupported format: {param}. '{type(offset).__name__}' is a non-fixed frequency. "
                        f"Please use a fixed frequency like '1s', '5min', '0.5h', '1d', '1d1h'."
                    )
            except ValueError:
                raise ValueError(
                    f"Invalid {param_name} format: {param}. Accepted formats are for example: '1s', '5min', '0.5h', '1d', '1d1h'."
                )
        elif allow_nan and isinstance(param, float) and np.isnan(param):
            pass  # Allow NaN if specified
        else:
            raise ValueError(
                f"{param_name} must be one of the following: "
                "a positive number, a valid time-based string (e.g., '1s', '5min', '1h'), "
                f"or {'NaN' if allow_nan else 'not NaN'}."
            )
            
    def _execute_dask(self, grouped_data, feature_columns):
        """
        Execute feature extraction using Dask.
        """
        dask_tasks = []

        for _, group in grouped_data:
            group_ddf = dd.from_pandas(group, npartitions=max(1, len(group) // 1000))
            # window_size = self.window_size if not pd.isna(self.window_size) else len(group)
            window_size = self._convert_window_to_observations(self.window_size, group) if not pd.isna(self.window_size) else len(group)
            stride = self._convert_window_to_observations(self.stride, group)

            if window_size > len(group):
                print(f"Warning: Window size ({window_size}) exceeds group length ({len(group)}). Skipping group.")
                continue

            meta = pd.DataFrame(columns=[f"{feature}_{col}" for feature in self.features for col in feature_columns])

            dask_tasks.append(
                group_ddf.map_partitions(
                    lambda partition: self._process_partition(partition, feature_columns, window_size, stride),
                    meta=meta
            ))

        with ProgressBar():
            dask_result = dd.concat(dask_tasks).compute()

        return dask_result

    def _process_partition(self, partition, feature_columns, window_size, stride):
        """
        Process a single partition of data to calculate features.

        Parameters
        ----------
        partition : pd.DataFrame
            The partition of data to process.
        feature_columns : list of str
            The columns of the partition to process.
        window_size : int
            The size of the window for feature extraction.

        Returns
        -------
        pd.DataFrame
            A DataFrame with calculated features for each partition.
        """
        results = []

        for start in range(0, len(partition) - window_size + 1, stride):
            window = partition.iloc[start : start + window_size]
            results.append(self._process_window(window, feature_columns))

        return pd.DataFrame(results)
    
    def _generate_tasks(self, grouped_data, feature_columns):
        """
        Generate feature extraction tasks for all groups and windows.
        """
        tasks = []
        for _, group in grouped_data:
            group_length = len(group)
            window_size = group_length if pd.isna(self.window_size) else self._convert_window_to_observations(self.window_size, group)
            stride = self._convert_window_to_observations(self.stride, group)

            if window_size > group_length:
                print(f"Warning: Window size ({window_size}) exceeds group length ({group_length}). Skipping group.")
                continue

            for start in range(0, group_length - window_size + 1, stride):
                window = group.iloc[start : start + window_size]
                tasks.append((window, feature_columns))
        return tasks
          
    def _execute_parallel(self, tasks, n_jobs, progress_callback, total_steps):
        """
        Execute feature extraction in parallel mode.
        """
        results = []
        completed_steps = 0

        def update_progress():
            nonlocal completed_steps
            completed_steps += 1
            if progress_callback:
                progress_callback(int((completed_steps / total_steps) * 100))

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_window_with_progress)(task, update_progress) for task in tasks
        )
        return results
        
    def _execute_sequential(self, tasks, progress_callback, total_steps):
        """
        Execute feature extraction in sequential mode.
        """
        results = []
        for completed_steps, (window, feature_columns) in enumerate(tasks, 1):
            results.append(self._process_window(window, feature_columns))
            if progress_callback:
                progress = int((completed_steps / total_steps) * 100)
                progress_callback(progress)
        return results
        
    def _process_window_with_progress(self, task, progress_callback):
        """
        Process a single window and report progress.
        """
        window, feature_columns = task
        result = self._process_window(window, feature_columns)
        progress_callback()
        return result

    def _process_window(self, window, feature_columns):
        """
        Process a single window to calculate features.

        Parameters
        ----------
        window : pd.DataFrame
            The window of data to process.
        feature_columns : list of str
            The columns of the window to process.

        Returns
        -------
        dict
            A dictionary of calculated features.
        """
        extracted_features = {}
        for feature_name in self.features:
            params = self.feature_params.get(feature_name, {})
            for col in feature_columns:
                try:
                    feature_data = window[col]
                    
                    self._validate_feature_data(feature_name, feature_data)
                    
                    if feature_data.empty:
                        extracted_features[f"{feature_name}_{col}"] = pd.NA
                    else:
                        extracted_features[f"{feature_name}_{col}"] = self._calculate_feature(feature_name, feature_data, params)
                except Exception as e:
                    warning_key = f"{feature_name}_{col}_{str(e)}"
                    if warning_key not in self.warning_registry:
                        print(f"Warning: Failed to calculate {feature_name} for column {col}: {e}")
                        self.warning_registry.add(warning_key)

                    extracted_features[f"{feature_name}_{col}"] = pd.NA
                    
        return extracted_features

    def _validate_feature_data(self, feature_name, data):
        """
        Validate data for a specific feature based on its requirements.
        """
        requirements = self.validation_requirements.get(feature_name, {'allow_nan': False, 'require_datetime_index': False})
        validate_time_series_data(data, require_datetime_index=requirements['require_datetime_index'], allow_nan=requirements['allow_nan'])

    def _convert_window_to_observations(self, value, data=None):
        """
        Converts a symbolic time value into a number of observations, based on the data frequency.
        """
        if isinstance(value, (int)):
            return value  # Return numeric values directly

        if isinstance(value, str):
            # Convert the symbolic time to a number of observations
            offset = to_offset(value)
            freq_in_nanos = pd.to_timedelta(data.index.freq).value
            return max(1, offset.nanos // freq_in_nanos)

        raise ValueError(f"Invalid window size or stride value: {value}")