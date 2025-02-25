import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from ..utils.feature_loader import Features
from ..utils.data_manager import load_metadata, load_feature_functions, load_validation_requirements
from ..utils.task_manager import TaskManager

class FeatureExtractor:
    DEFAULT_FEATURES_SMALL = [
        Features.LENGTH, Features.MEAN, Features.VARIANCE, Features.STABILITY,
        Features.ENTROPY, Features.SPIKENESS, Features.SEASONALITY_STRENGTH
    ]
    
    DEFAULT_FEATURES_BIG = [
    Features.ABSOLUTE_ENERGY, Features.BINARIZE_MEAN,
    Features.CHANGE_IN_VARIANCE, Features.CROSSING_POINTS, Features.DISTANCE_TO_LAST_TREND_CHANGE, Features.DOMINANT,
    Features.ENTROPY, Features.FLAT_SPOTS, Features.HETEROGENEITY,
    Features.LINEARITY, Features.LENGTH, Features.MEAN, Features.MISSING_POINTS,
    Features.PEAK, Features.SIGNIFICANT_CHANGES, Features.SPIKENESS, Features.STABILITY,
    Features.STD_1ST_DER, Features.TROUGH, Features.VARIANCE, Features.MEAN_CHANGE,
    Features.SEASONALITY_STRENGTH, Features.TREND_STRENGTH, Features.CHANGE_IN_VARIANCE
    ]
    
    FEATURES_ALL = [
    Features.ABOVE_9TH_DECILE, Features.BELOW_1ST_DECILE, Features.ABSOLUTE_ENERGY, Features.BINARIZE_MEAN,
    Features.CHANGE_IN_VARIANCE, Features.CROSSING_POINTS, Features.DISTANCE_TO_LAST_TREND_CHANGE, Features.DOMINANT,
    Features.ENTROPY, Features.FLAT_SPOTS, Features.HETEROGENEITY,
    Features.LINEARITY, Features.LENGTH, Features.MEAN, Features.MISSING_POINTS, Features.OUTLIERS_IQR, Features.OUTLIERS_STD,
    Features.PEAK, Features.SIGNIFICANT_CHANGES, Features.SPIKENESS, Features.STABILITY,
    Features.STD_1ST_DER, Features.TROUGH, Features.VARIANCE, Features.MEAN_CHANGE,
    Features.SEASONALITY_STRENGTH, Features.TREND_STRENGTH, Features.VARIABILITY_IN_SUB_PERIODS, Features.CHANGE_IN_VARIANCE
    ]

    FOR_ML = [
    Features.ABSOLUTE_ENERGY,Features.BINARIZE_MEAN,Features.DOMINANT,Features.ENTROPY,Features.FLAT_SPOTS,
    Features.HETEROGENEITY,Features.LINEARITY,Features.LENGTH,Features.MEAN,Features.MISSING_POINTS,Features.PEAK,
    Features.SIGNIFICANT_CHANGES,Features.SPIKENESS,Features.STABILITY,Features.STD_1ST_DER,Features.TROUGH,Features.VARIANCE,
    Features.SEASONALITY_STRENGTH, Features.TREND_STRENGTH
    ]
    
    CAN_USE_NAN = [
    Features.MISSING_POINTS, Features.PEAK, Features.SPIKENESS, Features.TROUGH, Features.SEASONALITY_STRENGTH
    ]
    
    def __init__(self, features=None, feature_params=None, window_size=np.nan, stride=1, id_column=None, sort_column=None, feature_column=None, group_by=None):
        """
        Initialize the FeatureExtractor with a list of features to calculate and optional parameters for each feature.

        Parameters
        ----------
        features : list of Features constants or str, optional
            A list of features to calculate, or a keyword ('small', 'big', 'all', 'for-ML', 'can-use-nan', 'empty').
            Default is None, which calculates the small default feature set.
        feature_params : dict, optional
            Parameters for specific features, where keys are feature names and values are dicts of parameters.
        window_size : int or str, optional
            The size of the window for feature extraction. 
            - np.nan (entire series is used as a single window),
            - int (number of samples in the window),
            - str (time-based format).
            Default is np.nan.
            If a string is provided, it must follow a fixed frequency format. 
            Supported formats include:
                - '1s' for 1 second
                - '5min' for 5 minutes
                - '0.5h' for 30 minutes
                - '1h' for 1 hour
                - '1d' for 1 day
                - A combination of time units, e.g., '1h15min' for 1 hour and 15 minutes.
            Only fixed frequencies are allowed. The window size cannot be a non-fixed frequency, such as business days ('1B') or other irregular intervals.
        stride : int or str, optional
            The step size for moving the window. Can be:
            - int (number of samples to shift),
            - str (time-based format).
            Default is 1.
            If a string is provided, it must follow the same fixed frequency format as the `window_size`.
        id_column : str, optional
            The name of the column used to identify different time series (optional).
        sort_column : str, optional
            The column to sort by before feature extraction (optional).
        feature_column : str or None, optional
            The column containing feature data. If None, features are calculated for all columns except ID and sort columns.
        group_by : str or None, optional
            Column name to group by. If None, no grouping is performed.
        Raises
        -------
        ValueError
            If any parameter is invalid.
        """        
        self.group_by = group_by

        if isinstance(features, str):
            if features.lower() == 'default-small':
                self.features = self.DEFAULT_FEATURES_SMALL
            elif features.lower() == 'default-big':
                self.features = self.DEFAULT_FEATURES_BIG
            elif features.lower() == 'all':
                self.features = self.FEATURES_ALL
            elif features.lower() == 'for-ml':
                self.features = self.FOR_ML
            elif features.lower() == 'can-use-nan':
                self.features = self.FOR_ML
            elif features.lower() == 'empty':
                self.features = []
            else:
                raise ValueError(f"Invalid feature keyword '{features}'. Accepted values are: 'default-small', 'default-big', 'all', 'for-ML', 'can-use-nan', 'empty'.")
        else:
            self.features = features if features is not None else self.DEFAULT_FEATURES_SMALL

        self.feature_params = feature_params if feature_params is not None else {}
        
        for feature_name, params in self.feature_params.items():
            if "window_size" in params:
                print(f"Warning: 'window_size' parameter in feature_params for feature '{feature_name}' will be ignored.")
                del params["window_size"]
            
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
        self.task_manager._validate_parameters(self.features, self.feature_params, self.window_size, self.stride, self.id_column, self.sort_column)
        self.feature_metadata = load_metadata()

    def validate_data_frequency(self, grouped_data):
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
            for _, group in grouped_data:
                if not isinstance(group.index, pd.DatetimeIndex):
                    raise ValueError(
                        "Time-based window_size and stride require a time-indexed DataFrame with regular frequency."
                    )                
                if group.index.freq is None:
                    if len(group) < 3:
                        inferred_freq = None
                    else:
                        inferred_freq = pd.infer_freq(group.index)
                    if inferred_freq is None:
                        raise ValueError(
                            "Data index does not have a defined frequency. Use `.resample()` to align your data."
                        )
                    group.index.freq = inferred_freq
    
    def head(self, features_df, n=5):
        """
        Returns the first n rows of the resulting DataFrame from the extract_features function.

        Parameters
        ----------
        features_df : pd.DataFrame
            The resulting DataFrame from the extract_features function.
        n : int, optional (default 5)
            The number of rows to return. If n is negative, returns all rows except the last |n| rows.

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
        """
        if mode not in ['parallel', 'sequential', 'dask']:
            raise ValueError(f"Invalid mode '{mode}'. Accepted values are: ['parallel', 'sequential']")

        if isinstance(data, pd.Series):
            data = data.to_frame(name='value')
            if self.feature_column is None:
                self.feature_column = 'value'

        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        if data.empty:
            print("Warning: Input data is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
        
        if self.sort_column:
            data = data.sort_values(by=self.sort_column)

        feature_columns = [self.feature_column] if self.feature_column else [col for col in data.columns if col not in {self.id_column, self.sort_column}]
        grouped_data = data.groupby(self.id_column) if self.id_column else [(None, data)]
        grouped_data = data.groupby(self.id_column) if self.id_column else [(None, data)]

        # TODO
        # grouped_data = self.group_data(data)

        self.validate_data_frequency(grouped_data)

        if mode == 'dask':
            return self.task_manager._execute_dask(grouped_data, feature_columns)

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
            
            data_stream = iter(data_stream)
        
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

    def add_custom_feature(self, name, function, metadata=None, params=None):
        """
        Add a custom feature to the FeatureExtractor with optional parameters.

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
        params : dict, optional
            A dictionary of parameters to be passed to the feature function when it is executed.
            Example:
            {
                'level': 'easy' | 'moderate' | 'advanced',
                'description': 'Description of the feature.'
            }


        Raises
        ------
        ValueError
            If the feature name already exists or the function is not callable.
        """
        if not hasattr(self, '_local_feature_functions'):
            self.feature_functions = self.feature_functions.copy()
            self.features = list(self.features)
            self.feature_metadata = self.feature_metadata.copy()
            self.feature_params = self.feature_params.copy()

        if name in self.feature_functions:
            raise ValueError(f"Feature '{name}' already exists.")
        if not callable(function):
            raise ValueError("The provided function is not callable.")

        self.feature_functions[name] = function
        self.features.append(name)
        self.feature_params[name] = params or {}

        if metadata:
            if 'level' not in metadata or 'description' not in metadata:
                raise ValueError("Metadata must include 'level' and 'description'.")
            self.feature_metadata[name] = metadata

        print(f"Custom feature '{name}' added successfully.")
