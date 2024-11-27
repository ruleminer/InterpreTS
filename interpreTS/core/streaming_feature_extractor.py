import pandas as pd
import numpy as np

from .features.feature_crossing_points import calculate_crossing_points
from .features.feature_spikeness import calculate_spikeness
from .features.feature_peak import calculate_peak
from .features.feature_trough import calculate_trough
from .features.feature_length import calculate_length
from .features.feature_mean import calculate_mean
from .features.seasonality_strength import calculate_seasonality_strength
from .features.feature_variance import calculate_variance
from .features.feature_std_1st_der import calculate_std_1st_der
from .features.feature_flat_spots import calculate_flat_spots

class StreamingFeatureExtractor:
    def __init__(self, features=None, feature_params=None, window_size=5):
        """
        Initialize the StreamingFeatureExtractor with a list of features to calculate and optional parameters.
        
        Parameters
        ----------
        features : list of str, optional
            A list of features to calculate. Default is None, which calculates all available features.
        feature_params : dict, optional
            Parameters for specific features, where keys are feature names and values are dicts of parameters.
        window_size : int, optional
            The size of the window for feature extraction.
        """
        self.features = features if features is not None else [
            'length', 'mean', 'peak', 'std_1st_der', 'trough', 
            'variance', 'spikeness', 'seasonality_strength', 'flat_spots', 
            'crossing_points'
        ]
        self.feature_params = feature_params if feature_params is not None else {}
        self.window_size = window_size
        self.data_buffer = []

        # Map of feature names to calculation functions
        self.feature_functions = {
            'length': calculate_length,
            'mean': calculate_mean,
            'peak': calculate_peak,
            'std_1st_der': calculate_std_1st_der,
            'trough': calculate_trough,
            'variance': calculate_variance,
            'spikeness': calculate_spikeness,
            'seasonality_strength': calculate_seasonality_strength,
            'flat_spots': calculate_flat_spots,
            'crossing_points': calculate_crossing_points,
        }

    def add_data(self, new_data):
        """
        Add new data to the buffer and calculate features if the window size is met.
        
        Parameters
        ----------
        new_data : float or int
            New data point to add to the buffer.
            
        Returns
        -------
        dict or None
            A dictionary of calculated features if the window size is met, else None.
        """
        self.data_buffer.append(new_data)

        # Sprawdzanie długości bufora przed obliczaniem cech
        if len(self.data_buffer) < self.window_size:
            return None

        window_data = pd.Series(self.data_buffer[-self.window_size:])
        features = self._calculate_features(window_data)
        return features


    def _calculate_features(self, data):
        """
        Calculate selected features on the current window data.
        
        Parameters
        ----------
        data : pd.Series
            The data window on which to calculate features.
            
        Returns
        -------
        dict
            A dictionary containing calculated features.
        """
        extracted_features = {}

        for feature_name in self.features:
            if feature_name in self.feature_functions:
                # Retrieve parameters for each feature, if provided
                params = self.feature_params.get(feature_name, {})
                extracted_features[feature_name] = self.feature_functions[feature_name](data, **params)

        return extracted_features
