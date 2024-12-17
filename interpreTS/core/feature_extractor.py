import pandas as pd
import numpy as np

from .features.feature_spikeness import calculate_spikeness
from .features.feature_entropy import calculate_entropy
from .features.feature_stability import calculate_stability
from .features.feature_length import calculate_length
from .features.feature_mean import calculate_mean
from .features.seasonality_strength import calculate_seasonality_strength
from .features.feature_variance import calculate_variance
from .features.feature_peak import calculate_peak
from .features.feature_trough import calculate_trough
from .features.feature_heterogeneity import heterogeneity
from .features.feature_absolute_energy import absolute_energy
from .features.feature_missing_points import missing_points
from .features.distance_to_the_last_change_point import calculate_distance_to_last_trend_change
from .features.feature_above_9th_decile import calculate_above_9th_decile
from .features.feature_below_1st_decile import calculate_below_1st_decile
from .features.feature_binarize_mean import calculate_binarize_mean
from .features.feature_crossing_points import calculate_crossing_points
from .features.feature_flat_spots import calculate_flat_spots
from .features.feature_outliers_iqr import calculate_outliers_iqr
from .features.feature_outliers_std import calculate_outliers_std
# from .features.feature_std_1st_der import calculate_std_1st_der
# from .features.feature_significant_changes import calculate_significant_changes

class Features:
    LENGTH = 'length'
    MEAN = 'mean'
    VARIANCE = 'variance'
    SPIKENESS = 'spikeness'
    ENTROPY = 'entropy'
    SEASONALITY_STRENGTH = 'seasonality_strength'
    STABILITY = 'stability'
    PEAK = 'peak'
    TROUGH = 'trough'
    DISTANCE_TO_LAST_TREND_CHANGE = 'distance_to_last_trend_change'
    HETEROGENEITY = 'heterogeneity'
    ABSOLUTE_ENERGY = 'absolute_energy'
    MISSING_POINTS = 'missing_points'
    ABOVE_9TH_DECILE = 'above_9th_decile'
    BELOW_1ST_DECILE = 'below_1st_decile'
    BINARIZE_MEAN = 'binarize_mean'
    CROSSING_POINTS = 'crossing_points'
    FLAT_SPOTS = 'flat_spots'
    OUTLIERS_IQR = 'outliers_iqr'
    OUTLIERS_STD = 'outliers_std'
    # STD_1ST_DER = 'std_1st_der'
    # SIGNIFICANT_CHANGES = 'significant_changes'

class FeatureExtractor:
    DEFAULT_FEATURES = [
        Features.LENGTH, Features.MEAN, Features.VARIANCE, Features.STABILITY,
        Features.ENTROPY, Features.SPIKENESS, Features.SEASONALITY_STRENGTH
    ]

    def __init__(self, features=None, feature_params=None, window_size=np.nan, stride=1, id_column=None, sort_column=None, feature_column=None):
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
        Raises
        -------
        ValueError
            If any parameter is invalid.
        """
        self._validate_parameters(features, feature_params, window_size, stride, id_column, sort_column)

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
            Features.HETEROGENEITY: heterogeneity,
            Features.ABSOLUTE_ENERGY: absolute_energy,
            Features.MISSING_POINTS: missing_points,
            Features.ABOVE_9TH_DECILE: calculate_above_9th_decile,
            Features.BELOW_1ST_DECILE: calculate_below_1st_decile,
            Features.BINARIZE_MEAN: calculate_binarize_mean,
            Features.CROSSING_POINTS: calculate_crossing_points,
            Features.FLAT_SPOTS: calculate_flat_spots,
            Features.OUTLIERS_IQR: calculate_outliers_iqr,
            Features.OUTLIERS_STD: calculate_outliers_std,
            # Features.STD_1ST_DER: calculate_std_1st_der,
            # Features.SIGNIFICANT_CHANGES: calculate_significant_changes,
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
            }
            # Features.STD_1ST_DER: {
            #     'level': 'moderate',
            #     'description': 'Standard deviation of the first derivative of the signal.'
            # },
            # Features.MISSING_POINTS: {
            #     'level': 'easy',
            #     'description': 'Proportion or count of missing data points in the window.'
            # },

        }

    def head(self, features_df, n=5):
        """
        Zwraca pierwsze n wierszy wynikowego DataFrame z funkcji extract_features.

        Parametry
        ----------
        features_df : pd.DataFrame
            DataFrame wynikowy z funkcji extract_features.
        n : int, opcjonalne (domyślnie 5)
            Liczba wierszy do zwrócenia. Jeśli n jest ujemne, zwraca wszystkie wiersze oprócz ostatnich |n| wierszy.

        Zwraca
        -------
        pd.DataFrame
            Pierwsze n wierszy DataFrame.
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

    def extract_features(self, data):
        """
        Extract features from a time series dataset.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            The time series data for which features are to be extracted.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing calculated features for each window.
        """
        if data.empty:
            print("Warning: Input data is empty. Returning an empty DataFrame.")
            return pd.DataFrame()

        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.feature_column)

        if self.sort_column:
            data = data.sort_values(by=self.sort_column)

        # Wyklucz kolumny ID i sort_column, jeśli feature_column=None
        feature_columns = (
            [self.feature_column] if self.feature_column else
            [col for col in data.columns if col not in {self.id_column, self.sort_column}]
        )

        grouped_data = data.groupby(self.id_column) if self.id_column else [(None, data)]

        results = []
        for _, group in grouped_data:
            group_length = len(group)
            window_size = group_length if np.isnan(self.window_size) else int(self.window_size)

            if window_size > group_length:
                print(f"Warning: Window size ({window_size}) exceeds group length ({group_length}). Skipping group.")
                continue

            for start in range(0, group_length - window_size + 1, self.stride):
                window = group.iloc[start:start + window_size]
                extracted_features = {self.id_column: group[self.id_column].iloc[0]} if self.id_column else {}

                for feature_name in self.features:
                    if feature_name in self.feature_functions:
                        params = self.feature_params.get(feature_name, {})
                        for col in feature_columns:
                            try:
                                feature_data = window[col].dropna()

                                # Walidacja danych przed obliczeniami
                                if feature_data.empty:
                                    extracted_features[f"{feature_name}_{col}"] = np.nan
                                    continue
                                if feature_data.nunique() == 1:
                                    # Stałe dane, np. wszystkie wartości są takie same
                                    extracted_features[f"{feature_name}_{col}"] = np.nan
                                    continue
                                if len(feature_data) < 2:
                                    # Za mało danych, aby wykonać obliczenia
                                    extracted_features[f"{feature_name}_{col}"] = np.nan
                                    continue

                                # Wywołanie funkcji cechy
                                extracted_features[f"{feature_name}_{col}"] = self.feature_functions[feature_name](
                                    feature_data, **params
                                )
                            except Exception as e:
                                print(f"Warning: Failed to calculate {feature_name} for column {col}: {e}")
                                extracted_features[f"{feature_name}_{col}"] = np.nan

                results.append(extracted_features)

        if not results:
            print("Warning: No features could be extracted. Returning an empty DataFrame.")
            return pd.DataFrame()

        return pd.DataFrame(results)
    
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
