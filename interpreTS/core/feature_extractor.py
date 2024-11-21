import pandas as pd
import numpy as np

from .features.feature_binarize_mean import calculate_binarize_mean
from .features.feature_crossing_points import calculate_crossing_points
from .features.feature_spikeness import calculate_spikeness
from .features.feature_peak import calculate_peak
from .features.feature_trough import calculate_trough
from .features.feature_length import calculate_length
from .features.feature_mean import calculate_mean
from .features.seasonality_strength import calculate_seasonality_strength
from .features.feature_variance import calculate_variance
from .features.feature_std_1st_der import calculate_std_1st_der
from .features.feature_entropy import calculate_entropy
from .features.feature_stability import calculate_stability
from .features.feature_flat_spots import calculate_flat_spots

class Features:
    LENGTH = 'length'
    MEAN = 'mean'
    PEAK = 'peak'
    STD_1ST_DER = 'std_1st_der'
    TROUGH = 'trough'
    VARIANCE = 'variance'
    SPIKENESS = 'spikeness'
    ENTROPY = 'entropy'
    CALCULATE_SEASONALITY_STRENGTH = 'seasonality_strength'
    STABILITY = 'stability'
    FLAT_SPOTS = 'flat_spots'
    CROSSING_POINTS = 'crossing_points'
    BINARIZE_MEAN = 'binarize_mean'


class FeatureExtractor:
    def __init__(self, features=None, feature_params=None, window_size=5, stride=1, id_column=None, sort_column=None):
        """
        Initialize the FeatureExtractor with a list of features to calculate and optional parameters for each feature.
        
        Parameters
        ----------
        features : list of Features constants, optional
            A list of features to calculate. Default is None, which calculates all available features.
        feature_params : dict, optional
            Parameters for specific features, where keys are feature names and values are dicts of parameters.
        window_size : int, optional
            The size of the window for feature extraction (default is 5).
        stride : int, optional
            The step size for moving the window (default is 1).
        id_column : str, optional
            The name of the column used to identify different time series (optional).
        sort_column : str, optional
            The column to sort by before feature extraction (optional).
        """
        
        self.features = features if features is not None else [Features.LENGTH]
        self.feature_params = feature_params if feature_params is not None else {}
        self.window_size = window_size
        self.stride = stride
        self.id_column = id_column
        self.sort_column = sort_column

        # Map of feature names to calculation functions
        self.feature_functions = {
            Features.LENGTH: calculate_length,
            Features.MEAN: calculate_mean,
            Features.PEAK: calculate_peak,
            Features.STD_1ST_DER: calculate_std_1st_der,
            Features.TROUGH: calculate_trough,
            Features.VARIANCE: calculate_variance,
            Features.SPIKENESS: calculate_spikeness,
            Features.CALCULATE_SEASONALITY_STRENGTH: calculate_seasonality_strength,
            Features.ENTROPY: calculate_entropy,
            Features.STABILITY: calculate_stability,
            Features.FLAT_SPOTS: calculate_flat_spots, 
            Features.CROSSING_POINTS: calculate_crossing_points, 
            Features.BINARIZE_MEAN: calculate_binarize_mean,
        }

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
        # Convert Series to DataFrame with 'value' column if necessary
        if isinstance(data, pd.Series):
            data = data.to_frame(name='value')

        # Sort data if sort_column is provided
        if self.sort_column:
            data = data.sort_values(by=self.sort_column)

        # Group by id_column if provided``
        grouped_data = data.groupby(self.id_column) if self.id_column else [(None, data)]

        results = []

        # Process each group separately
        for _, group in grouped_data:
            for start in range(0, len(group) - self.window_size + 1, self.stride):
                window = group.iloc[start:start + self.window_size]
                extracted_features = {self.id_column: group[self.id_column].iloc[0]} if self.id_column else {}

                # Calculate each selected feature on the window
                for feature_name in self.features:
                    if feature_name in self.feature_functions:
                        params = self.feature_params.get(feature_name, {})
                        feature_data = window['value'] if 'value' in window else window.iloc[:, 0]
                        extracted_features[feature_name] = self.feature_functions[feature_name](feature_data, **params)

                results.append(extracted_features)

        return pd.DataFrame(results) 

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
