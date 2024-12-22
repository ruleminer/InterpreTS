import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed

from .features.feature_spikeness import calculate_spikeness
from .features.feature_entropy import calculate_entropy
from .features.feature_stability import calculate_stability
from .features.feature_length import calculate_length
from .features.feature_mean import calculate_mean
from .features.seasonality_strength import calculate_seasonality_strength
from .features.feature_variance import calculate_variance
from .features.feature_peak import calculate_peak
from .features.feature_trough import calculate_trough
from .features.feature_heterogeneity import calculate_heterogeneity
from .features.feature_absolute_energy import calculate_absolute_energy
from .features.feature_missing_points import calculate_missing_points
from .features.distance_to_the_last_change_point import calculate_distance_to_last_trend_change
from .features.feature_above_9th_decile import calculate_above_9th_decile
from .features.feature_below_1st_decile import calculate_below_1st_decile
from .features.feature_binarize_mean import calculate_binarize_mean
from .features.feature_crossing_points import calculate_crossing_points
from .features.feature_flat_spots import calculate_flat_spots
from .features.feature_outliers_iqr import calculate_outliers_iqr
from .features.feature_outliers_std import calculate_outliers_std
from .features.feature_std_1st_der import calculate_std_1st_der
from .features.histogram_dominant import calculate_dominant
from .features.mean_change import calculate_mean_change
from .features.trend_strength import calculate_trend_strength
from .features.feature_significant_changes import calculate_significant_changes
from .features.variability_in_sub_periods import calculate_variability_in_sub_periods
from .features.variance_change import calculate_change_in_variance
from .features.feature_linearity import calculate_linearity

class Features:
    LENGTH = 'length'
    MEAN = 'mean'
    DOMINANT = 'dominant'
    TREND_STRENGTH = 'trend_strength'
    SEASONALITY_STRENGTH = 'seasonality_strength'
    PEAK = 'peak'
    TROUGH = 'trough'
    SPIKENESS = 'spikeness'
    VARIANCE = 'variance'
    STABILITY = 'stability'
    FLAT_SPOTS = 'flat_spots'
    STD_1ST_DER = 'std_1st_der'
    CROSSING_POINTS = 'crossing_points'
    HETEROGENEITY = 'heterogeneity'
    LINEARITY = 'linearity'
    ENTROPY = 'entropy'
    VARIABILITY_IN_SUB_PERIODS = 'variability_in_sub_periods'
    OUTLIERS_STD = 'outliers_std'
    OUTLIERS_IQR = 'outliers_iqr'
    CHANGE_IN_VARIANCE = 'change_in_variance'
    MEAN_CHANGE = 'mean_change'
    SIGNIFICANT_CHANGES = 'significant_changes'
    MISSING_POINTS = 'missing_points'
    DISTANCE_TO_LAST_TREND_CHANGE = 'distance_to_last_trend_change'
    ABOVE_9TH_DECILE = 'above_9th_decile'
    BELOW_1ST_DECILE = 'below_1st_decile'
    ABSOLUTE_ENERGY = 'absolute_energy'
    BINARIZE_MEAN = 'binarize_mean'
    
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
        window_size : int or float (NaN), optional
            The size of the window for feature extraction. If NaN (default), the entire series is used as a single window.
        stride : int, optional
            The step size for moving the window (default is 1).
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
        self._validate_parameters(features, feature_params, window_size, stride, id_column, sort_column)
        self.group_by = group_by
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.feature_params = feature_params if feature_params is not None else {}
        self.window_size = window_size
        self.stride = stride
        self.id_column = id_column
        self.sort_column = sort_column
        self.feature_column = feature_column

        self.feature_functions = {
            Features.LENGTH: calculate_length,
            Features.MEAN: calculate_mean,
            Features.VARIANCE: calculate_variance,
            Features.SPIKENESS: calculate_spikeness,
            Features.ENTROPY: calculate_entropy,
            Features.STABILITY: calculate_stability,
            Features.SEASONALITY_STRENGTH: calculate_seasonality_strength,
            Features.PEAK: calculate_peak,
            Features.TROUGH: calculate_trough,
            Features.DISTANCE_TO_LAST_TREND_CHANGE: calculate_distance_to_last_trend_change,
            Features.HETEROGENEITY: calculate_heterogeneity,
            Features.ABSOLUTE_ENERGY: calculate_absolute_energy,
            Features.MISSING_POINTS: calculate_missing_points,
            Features.ABOVE_9TH_DECILE: calculate_above_9th_decile,
            Features.BELOW_1ST_DECILE: calculate_below_1st_decile,
            Features.BINARIZE_MEAN: calculate_binarize_mean,
            Features.CROSSING_POINTS: calculate_crossing_points,
            Features.FLAT_SPOTS: calculate_flat_spots,
            Features.OUTLIERS_IQR: calculate_outliers_iqr,
            Features.OUTLIERS_STD: calculate_outliers_std,
            Features.STD_1ST_DER: calculate_std_1st_der,
            Features.DOMINANT: calculate_dominant,
            Features.MEAN_CHANGE: calculate_mean_change,
            Features.TREND_STRENGTH: calculate_trend_strength,
            Features.SIGNIFICANT_CHANGES: calculate_significant_changes,
            Features.VARIABILITY_IN_SUB_PERIODS: calculate_variability_in_sub_periods,
            Features.CHANGE_IN_VARIANCE: calculate_change_in_variance,
            Features.LINEARITY: calculate_linearity
        }

        self.feature_metadata = {
            Features.LENGTH: {
                'level': 'easy',
                'description': 'Number of points in the window.'
            },
            Features.MEAN: {
                'level': 'easy',
                'description': 'Mean value within the window.'
            },
            Features.VARIANCE: {
                'level': 'moderate',
                'description': 'Variance of the signal within the window.'
            },
            Features.ENTROPY: {
                'level': 'advanced',
                'description': 'Degree of randomness or disorder in the window.'
            },
            Features.SPIKENESS: {
                'level': 'moderate',
                'description': 'Measure of sudden jumps or spikes in the signal.'
            },
            Features.SEASONALITY_STRENGTH: {
                'level': 'advanced',
                'description': 'Strength of seasonal patterns within the signal.'
            },
            Features.STABILITY: {
                'level': 'moderate',
                'description': 'Measure of consistency in the signal values.'
            },
            Features.PEAK: {
                 'level': 'easy',
                 'description': 'The maximum value in the window.'
            },
            Features.TROUGH: {
                 'level': 'easy',
                 'description': 'The minimum value in the window.'
            },
            Features.DISTANCE_TO_LAST_TREND_CHANGE: {
                'level': 'moderate',
                'description': 'Distance (in terms of indices) to the last detected trend change in the window.'
            },
            Features.ABSOLUTE_ENERGY: {
                 'level': 'moderate',
                 'description': 'Total energy of the signal in the window.'
             },
            Features.ABOVE_9TH_DECILE: {
                'level': 'moderate',
                'description': 'Fraction of values in the window above the 9th decile of the training data, representing the presence of extreme high values.'
            },
            Features.BELOW_1ST_DECILE: {
                'level': 'moderate',
                'description': 'Fraction of values in the window below the 1st decile of the training data, representing the presence of extreme low values.'
            },
            Features.BINARIZE_MEAN: {
                'level': 'moderate',
                'description': 'Binary value indicating whether the signal mean exceeds a threshold.'
            },
            Features.CROSSING_POINTS: {
                'level': 'easy',
                'description': 'Number of times the signal crosses its mean.'
            },
            Features.FLAT_SPOTS: {
                'level': 'easy',
                'description': 'Number of segments with constant values in the signal.'
            },
            Features.HETEROGENEITY: {
                'level': 'moderate',
                'description': 'Coefficient of variation, representing the ratio of standard deviation to mean, indicating the relative variability in the time series.'
            },
            Features.OUTLIERS_IQR: {
                'level': 'moderate',
                'description': 'Percentage of values in the window that are classified as outliers based on the Interquartile Range (IQR) method.'
            },
            Features.OUTLIERS_STD: {
                'level': 'moderate',
                'description': 'Percentage of values in the window that are more than 3 standard deviations away from the mean, indicating extreme deviations.'
            },
            Features.STD_1ST_DER: {
                'level': 'moderate',
                'description': 'Standard deviation of the first derivative of the signal.'
            },
            Features.DOMINANT: {
                'level': 'moderate',
                'description': 'The dominant value of the time series histogram, representing the most frequent range of values within the specified bins.'
            },
            Features.MEAN_CHANGE: {
                'level': 'moderate',
                'description': 'The rate of change in the rolling mean over time, capturing trends or shifts in the time series.'
            },
            Features.TREND_STRENGTH: {
                'level': 'moderate',
                'description': 'The R-squared value from a linear regression, representing the strength and consistency of the trend in the time series.'
            },
            Features.SIGNIFICANT_CHANGES: {
                'level': 'moderate',
                'description': 'Proportion of significant increases or decreases in the time series, based on deviations from the interquartile range (IQR) of differences between consecutive values.'
            },
            Features.MISSING_POINTS: {
                'level': 'easy',
                'description': 'Proportion or count of missing data points in the window.'
            },
            Features.VARIABILITY_IN_SUB_PERIODS: {
                'level': 'moderate',
                'description': 'Variance calculated within sub-periods of a time series, providing a measure of variability across fixed-size windows.'
            },
            Features.CHANGE_IN_VARIANCE: {
                'level': 'moderate',
                'description': 'Change in variance over time, calculated as the difference between rolling variances across consecutive windows.'
            },
            Features.LINEARITY:{
                'level': 'moderate',
                'description': 'Measure of how well the time series can be approximated by a linear trend, quantified using the R-squared value from linear regression.'
            }
        }
        
    def _calculate_feature(self, feature_name, feature_data, params):
        """
        Calculate a specific feature.
        """
        if feature_name in self.feature_functions:
            return self.feature_functions[feature_name](feature_data, **params)
        else:
            raise ValueError(f"Feature '{feature_name}' is not supported.")
    
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

    @staticmethod
    def _validate_parameters(features, feature_params, window_size, stride, id_column, sort_column):
        """
        Validate input parameters to ensure they are valid.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        available_features = FeatureExtractor.available_features()

        if features is not None:
            if not isinstance(features, list):
                raise ValueError("Features must be a list or None.")
            invalid_features = [feature for feature in features if feature not in available_features]
            if invalid_features:
                raise ValueError(
                    f"The following features are invalid or not implemented: {invalid_features}. "
                    f"Available features are: {available_features}."
                )
        if feature_params is not None and not isinstance(feature_params, dict):
            raise ValueError("Feature parameters must be a dictionary or None.")
        if not (np.isnan(window_size) or (isinstance(window_size, (int, float)) and window_size > 0)):
            raise ValueError("Window size must be a positive number or NaN.")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("Stride must be a positive integer.")
        if id_column is not None and not isinstance(id_column, str):
            raise ValueError("ID column must be a string or None.")
        if sort_column is not None and not isinstance(sort_column, str):
            raise ValueError("Sort column must be a string or None.")

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

        if data.empty:
            print("Warning: Input data is empty. Returning an empty DataFrame.")
            return pd.DataFrame()

        if self.sort_column:
            data = data.sort_values(by=self.sort_column)

        feature_columns = [self.feature_column] if self.feature_column else [col for col in data.columns if col not in {self.id_column, self.sort_column}]
        grouped_data = self.group_data(data)

        if mode == 'dask':
            return self._execute_dask(grouped_data, feature_columns, progress_callback)

        # Generate tasks for feature extraction
        tasks = self._generate_tasks(grouped_data, feature_columns)
        total_steps = len(tasks)

        # Execute in parallel or sequential mode
        if mode == 'parallel':
            results = self._execute_parallel(tasks, n_jobs, progress_callback, total_steps)
        else:
            results = self._execute_sequential(tasks, progress_callback, total_steps)

        return pd.DataFrame(results)
        
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

        # Initialize buffers for each time series ID
        buffers = {}
        total_points = 0

        for new_point in data_stream:
            total_points += 1
            series_id = new_point[self.id_column]

            # Initialize buffer if necessary
            if series_id not in buffers:
                buffers[series_id] = []

            # Add new point to the buffer
            buffers[series_id].append(new_point)

            # Keep only the most recent points within the window size
            if len(buffers[series_id]) > self.window_size:
                buffers[series_id].pop(0)

            # Calculate features if the buffer is full
            if len(buffers[series_id]) == self.window_size:
                buffer_df = pd.DataFrame(buffers[series_id])
                feature_columns = [self.feature_column]

                # Process the buffer and yield the result
                features = self._process_window(buffer_df, feature_columns)
                features[self.id_column] = series_id
                yield features

            # Report progress if a callback is provided
            if progress_callback:
                progress_callback(total_points)
            

    def _execute_dask(self, grouped_data, feature_columns, progress_callback):
        """
        Execute feature extraction using Dask.
        """
        dask_tasks = []

        for _, group in grouped_data:
            group_ddf = dd.from_pandas(group, npartitions=4)
            window_size = self.window_size if not pd.isna(self.window_size) else len(group)

            if window_size > len(group):
                print(f"Warning: Window size ({window_size}) exceeds group length ({len(group)}). Skipping group.")
                continue

            meta = pd.DataFrame(columns=[f"{feature}_{col}" for feature in self.features for col in feature_columns])

            dask_tasks.append(group_ddf.map_partitions(
                lambda partition: self._process_partition(partition, feature_columns, window_size),
                meta=meta
            ))

        # Combine results
        with ProgressBar():
            dask_result = dd.concat(dask_tasks).compute()

        return dask_result

    def _process_partition(self, partition, feature_columns, window_size):
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

        for start in range(0, len(partition) - window_size + 1, self.stride):
            window = partition.iloc[start : start + window_size]
            results.append(self._process_window(window, feature_columns))

        # Ensure the result is always a DataFrame
        return pd.DataFrame(results)
    
    def _generate_tasks(self, grouped_data, feature_columns):
        """
        Generate feature extraction tasks for all groups and windows.
        """
        tasks = []
        for _, group in grouped_data:
            group_length = len(group)
            window_size = group_length if pd.isna(self.window_size) else int(self.window_size)

            if window_size > group_length:
                print(f"Warning: Window size ({window_size}) exceeds group length ({group_length}). Skipping group.")
                continue

            for start in range(0, group_length - window_size + 1, self.stride):
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
                    feature_data = window[col].dropna()
                    if feature_data.empty:
                        extracted_features[f"{feature_name}_{col}"] = pd.NA
                    else:
                        extracted_features[f"{feature_name}_{col}"] = self._calculate_feature(feature_name, feature_data, params)
                except Exception as e:
                    print(f"Warning: Failed to calculate {feature_name} for column {col}: {e}")
                    extracted_features[f"{feature_name}_{col}"] = pd.NA
        return extracted_features

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
    
    def generate_feature_descriptions(self, extracted_features):
        """
        Generate textual descriptions for extracted features.

        Parameters
        ----------
        extracted_features : dict
            A dictionary where keys are feature names and values are their calculated values.

        Returns
        -------
        dict
            A dictionary where keys are feature names and values are textual descriptions.
        """
        descriptions = {}
        for feature_name, feature_value in extracted_features.items():
            if feature_name in self.feature_metadata:
                metadata = self.feature_metadata[feature_name]
                description = metadata['description']
                descriptions[feature_name] = (
                    f"Feature '{feature_name}': {description} Value: {feature_value}."
                )
            else:
                descriptions[feature_name] = (
                    f"Unknown feature: '{feature_name}'. Value: {feature_value}."
                )
        return descriptions

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
            A dictionary containing metadata about the feature (e.g., level of interpretability and description).
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
        if name in self.feature_functions:
            raise ValueError(f"Feature '{name}' already exists.")
        if not callable(function):
            raise ValueError("The provided function is not callable.")
        
        # Add the feature function
        self.feature_functions[name] = function
        self.features.append(name)
        # Add metadata if provided
        if metadata:
            if 'level' not in metadata or 'description' not in metadata:
                raise ValueError("Metadata must include 'level' and 'description'.")
            self.feature_metadata[name] = metadata

        print(f"Custom feature '{name}' added successfully.")


    @staticmethod
    def available_features():
        """
        Returns a list of all available features.

        Returns
        -------
        list
            List of feature names.
        """
        
        return list(Features.__dict__.values())


    def generate_feature_options(self):
        """
        Generate a dictionary mapping human-readable feature names to their corresponding constants.

        Returns
        -------
        dict
            A dictionary where keys are human-readable feature names (capitalized) 
            and values are feature constants.
        """
        available_features = self.available_features()

        feature_constants = {
            name: value
            for name, value in Features.__dict__.items()
            if not name.startswith('__') and not callable(value)
        }

        return {name.capitalize(): constant for name, constant in feature_constants.items()}
