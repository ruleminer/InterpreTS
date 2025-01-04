import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset

from ..utils.feature_loader import Features
from ..utils.data_manager import load_metadata, load_feature_functions, load_validation_requirements
from ..utils.task_manager import TaskManager

class FeatureExtractor:
    DEFAULT_FEATURES = [
        Features.LENGTH, Features.MEAN, Features.VARIANCE, Features.STABILITY,
        Features.ENTROPY, Features.SPIKENESS, Features.SEASONALITY_STRENGTH
    ]

    def __init__(self, features=None, feature_params=None, window_size=np.nan, stride=1, id_column=None, sort_column=None, feature_column=None, group_by=None):
        """
        Initialize the FeatureExtractor with a list of features to calculate and optional parameters for each feature.

        Parameters
        ----------
        features : list of Features constants, optional
            A list of features to calculate. Default is None, which calculates all available features.
        feature_params : dict, optional
            Parameters for specific features, where keys are feature names and values are dicts of parameters.
        window_size : int or str, optional
            The size of the window for feature extraction. 
            - np.nan (entire series is used as a single window),
            - int (number of samples in the window),
            - str (time-based format, e.g., '1s', '5min')
            Default is np.nan.
        stride : int or str, optional
            The step size for moving the window. Can be:
            - int (number of samples to shift),
            - str (time-based format, e.g., '1s', '5min').
            Default is 1.
        id_column : str, optional
            The name of the column used to identify different time series.
        sort_column : str, optional
            The column to sort by before feature extraction.
        feature_column : str or None, optional
            The column containing feature data. If None, features are calculated for all columns except ID and sort columns.
        group_by : str or None, optional
            Column name to group by. If None, no grouping is performed.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """        
        self.group_by = group_by
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.feature_params = feature_params if feature_params is not None else {}
        self.window_size = window_size
        self.stride = stride
        self.id_column = id_column
        self.sort_column = sort_column
        self.feature_column = feature_column

        self.feature_functions = load_feature_functions()
        self.validation_requirements = load_validation_requirements()

        self.task_manager = TaskManager(
            self.feature_functions, self.window_size, self.features, self.stride, 
            self.feature_params, self.validation_requirements
        )
        self.task_manager._validate_parameters(features, feature_params, window_size, stride, id_column, sort_column)
        self.feature_metadata = load_metadata()

    def validate_data_frequency(self, data):
        """
        Validate that data has a consistent and defined frequency if window_size or stride are time-based.

        Parameters
        ----------
        data : pd.DataFrame
            The time series data to validate.

        Raises
        ------
        ValueError
            If data frequency is not defined or inconsistent.
        """
        if isinstance(self.window_size, str) or isinstance(self.stride, str):
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(
                    "Time-based window_size and stride require a time-indexed DataFrame with regular frequency."
                )
            if data.index.freq is None:
                inferred_freq = pd.infer_freq(data.index)
                if inferred_freq is None:
                    raise ValueError(
                        "Data index does not have a defined frequency. Use `.resample()` to align your data."
                    )
                data.index.freq = inferred_freq
    
    def head(self, features_df, n=5):
        """
        Returns the first n rows of the resulting DataFrame from the extract_features function.

        Parameters
        ----------
        features_df : pd.DataFrame
            The resulting DataFrame from the extract_features function.
        n : int, optional
            The number of rows to return (default is 5). If n is negative, 
            returns all rows except the last abs(n) rows.

        Returns
        -------
        pd.DataFrame
            The first n rows of the DataFrame.
        """
        
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")
        if len(features_df) < n:
            print(f"Warning: Only {len(features_df)} rows available in DataFrame.")
        return features_df.head(n)
            
    def extract_features(self, data, progress_callback=None, mode='sequential', n_jobs=-1):
        """
        Extract features from a time series dataset.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            The time series data for which features are to be extracted.
        progress_callback : function, optional
            A function to report progress, which takes a single argument: progress percentage (0-100).
        mode : str, optional
            The mode of processing. Can be 'parallel' for multi-threaded processing
            or 'sequential' for single-threaded processing with real-time progress reporting.
        n_jobs : int, optional
            The number of jobs (processes) to run in parallel. Default is -1 (use all available CPUs).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing calculated features for each window.

        Raises
        ------
        ValueError
            If the mode is not one of ['parallel', 'sequential', 'dask'].
        """
        if mode not in ['parallel', 'sequential', 'dask']:
            raise ValueError(f"Invalid mode '{mode}'. Accepted values are: ['parallel', 'sequential']")

        if data.empty:
            print("Warning: Input data is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
        
        self.validate_data_frequency(data)

        if self.sort_column:
            data = data.sort_values(by=self.sort_column)

        feature_columns = [self.feature_column] if self.feature_column else [col for col in data.columns if col not in {self.id_column, self.sort_column}]
        grouped_data = self.group_data(data)

        if mode == 'dask':
            return self.task_manager._execute_dask(grouped_data, feature_columns, progress_callback)

        tasks = self.task_manager._generate_tasks(grouped_data, feature_columns)
        total_steps = len(tasks)

        if mode == 'parallel':
            results = self.task_manager._execute_parallel(tasks, n_jobs, progress_callback, total_steps)
        else:
            results = self.task_manager._execute_sequential(tasks, progress_callback, total_steps)

        return pd.DataFrame(results)
    
    def group_data(self, data):
        """
        Group data based on the group_by column.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.

        Returns
        -------
        iterable
            Grouped data.
        """
        if self.group_by:
            return data.groupby(self.group_by)
        return [(None, data)]
    
    def extract_features_stream(self, data_stream, progress_callback=None):
        """
        Extract features from a stream of time series data.

        Parameters
        ----------
        data_stream : iterable
            An iterable that yields incoming data points as dictionaries with keys corresponding to column names.
        progress_callback : function, optional
            A function to report progress, which takes a single argument: the total number of processed points.

        Yields
        ------
        dict
            A dictionary containing the calculated features for the current window.
        """
        if not self.feature_column or not self.id_column:
            raise ValueError("Feature column and ID column must be specified for streaming mode.")

        buffers = {}
        total_points = 0

        time_based_window = isinstance(self.window_size, str)

        if time_based_window:
            if self.sort_column is None:
                raise ValueError("A 'sort_column' must be specified when using a time-based window.")
        
            try:
                # Check if sort_column is datetime-based
                sample_point = next(data_stream)  # Get the first data point to check format
                if not pd.to_datetime(sample_point[self.sort_column], errors='coerce'):
                    raise ValueError(f"Column '{self.sort_column}' does not contain valid datetime values.")
                # Put the sample back into the generator stream
                data_stream = iter([sample_point] + list(data_stream))
            except Exception as e:
                raise ValueError(f"Error in validating time-based column: {e}")

            try:
                window_offset = to_offset(self.window_size)
            except ValueError:
                raise ValueError(f"Invalid time-based window_size format: {self.window_size}. Supported formats are for example: '1s', '5min', '1h'.")

        for new_point in data_stream:
            total_points += 1
            series_id = new_point[self.id_column]

            if series_id not in buffers:
                buffers[series_id] = []

            buffers[series_id].append(new_point)

            # Handle time-based windows
            if time_based_window:
                # Convert buffer to a DataFrame and check time range
                buffer_df = pd.DataFrame(buffers[series_id])
                if len(buffer_df) > 1:  # Ensure at least two points to calculate a range
                    start_time = pd.to_datetime(buffer_df[self.sort_column].iloc[0])
                    end_time = pd.to_datetime(buffer_df[self.sort_column].iloc[-1])
                    if (end_time - start_time) >= window_offset:
                        # Extract features for the current buffer
                        feature_columns = [self.feature_column]
                        features = self.task_manager._process_window(buffer_df, feature_columns)
                        features[self.id_column] = series_id
                        yield features
                        buffers[series_id] = buffers[series_id][1:]  # Remove oldest point

            # Handle numeric windows
            else:
                if len(buffers[series_id]) > self.window_size:
                    buffers[series_id].pop(0)

                if len(buffers[series_id]) == self.window_size:
                    buffer_df = pd.DataFrame(buffers[series_id])
                    feature_columns = [self.feature_column]

                    features = self.task_manager._process_window(buffer_df, feature_columns)
                    features[self.id_column] = series_id
                    yield features

            if progress_callback:
                progress_callback(total_points)
            
    def group_features_by_interpretability(self):
        """
        Group features by their interpretability levels.

        Returns
        -------
        dict
            A dictionary where keys are interpretability levels ('easy', 'moderate', 'advanced'),
            and values are lists of feature names.
        """
        groups = {'easy': [], 'moderate': [], 'advanced': []}
        for feature_name, metadata in self.feature_metadata.items():
            level = metadata['level']
            groups[level].append(feature_name)
        return groups

    def add_custom_feature(self, name, function, metadata=None):
        """
        Add a custom feature to the FeatureExtractor.

        Parameters
        ----------
        name : str
            The name of the custom feature.
        function : callable
            A function that computes the feature. It should accept a Pandas Series and optional parameters as input.
        metadata : dict, optional
            A dictionary containing metadata about the feature, such as its interpretability level and description.
            - level (str): Interpretability level ('easy', 'moderate', 'advanced').
            - description (str): Description of the feature.

        Raises
        ------
        ValueError
            If the feature name already exists or the function is not callable.
        """
        if name in self.feature_functions:
            raise ValueError(f"Feature '{name}' already exists.")
        if not callable(function):
            raise ValueError("The provided function is not callable.")
        
        self.feature_functions[name] = function
        self.features.append(name)

        if metadata:
            if 'level' not in metadata or 'description' not in metadata:
                raise ValueError("Metadata must include 'level' and 'description'.")
            self.feature_metadata[name] = metadata

        print(f"Custom feature '{name}' added successfully.")