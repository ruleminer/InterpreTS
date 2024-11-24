import pandas as pd
import numpy as np
from dask import delayed, compute
import ray

# Funkcje cech są importowane tak jak w `FeatureExtractor`
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
from .features.feature_absolute_energy import absolute_energy
from .features.feature_entropy import calculate_entropy
from .features.feature_stability import calculate_stability
from .features.feature_flat_spots import calculate_flat_spots
from .features.feature_missing_points import missing_points

class ParallelFeatureExtractor:
    def __init__(self, features=None, feature_params=None, window_size=5, stride=1, use_dask=False, use_ray=False):
        """
        Initialize the ParallelFeatureExtractor with a list of features and parallelization options.
        
        Parameters
        ----------
        features : list of str, optional
            A list of features to calculate. Default is None, which calculates all available features.
        feature_params : dict, optional
            Parameters for specific features, where keys are feature names and values are dicts of parameters.
        window_size : int, optional
            The size of the window for feature extraction (default is 5).
        stride : int, optional
            The step size for moving the window (default is 1).
        use_dask : bool, optional
            Whether to use Dask for parallel processing.
        use_ray : bool, optional
            Whether to use Ray for parallel processing.
        """
        self.features = features if features is not None else [
            'length', 'mean', 'peak', 'std_1st_der', 'trough', 
            'variance', 'spikeness', 'seasonality_strength', 'flat_spots', 
            'crossing_points', 'binarize_mean', 'entropy', 'stability', 'absolute_energy', 'missing_points'
        ]
        self.feature_params = feature_params if feature_params is not None else {}
        self.window_size = window_size
        self.stride = stride
        self.use_dask = use_dask
        self.use_ray = use_ray

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
            'binarize_mean': calculate_binarize_mean,
            'entropy': calculate_entropy,
            'stability': calculate_stability,
            'absolute_energy': absolute_energy,
            'missing_points': missing_points,
        }

        if self.use_ray:
            ray.init(ignore_reinit_error=True)

    def extract_features(self, data):
        """
        Extract features from a time series dataset with parallel processing.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            The time series data for which features are to be extracted.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing calculated features for each window.
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name='value')

        results = []

        # Sequential or grouped processing
        for start in range(0, len(data) - self.window_size + 1, self.stride):
            window = data.iloc[start:start + self.window_size]

            if self.use_dask:
                results.append(self._calculate_features_dask(window))
            elif self.use_ray:
                results.append(self._calculate_features_ray(window))
            else:
                results.append(self._calculate_features(window))

        if self.use_dask:
            results = compute(*results)

        elif self.use_ray:
            results = ray.get(results)

        return pd.DataFrame(results)

    def _calculate_features(self, window):
        """
        Calculate features for a single window sequentially.
        """
        extracted_features = {}
        for feature_name in self.features:
            if feature_name in self.feature_functions:
                params = self.feature_params.get(feature_name, {})
                extracted_features[feature_name] = self.feature_functions[feature_name](window['value'], **params)
        return extracted_features

    def _calculate_features_dask(self, window):
        """
        Calculate features for a single window using Dask.
        """
        return delayed(self._calculate_features)(window)

    def _calculate_features_ray(self, window):
        """
        Calculate features for a single window using Ray.
        """
        @ray.remote
        def calculate_single_feature(feature_function, data, params):
            return feature_function(data, **params)

        futures = []
        for feature_name in self.features:
            if feature_name in self.feature_functions:
                params = self.feature_params.get(feature_name, {})
                futures.append(
                    calculate_single_feature.remote(self.feature_functions[feature_name], window['value'], params)
                )
        return dict(zip(self.features, ray.get(futures)))

    @staticmethod
    def available_features():
        """
        Returns a list of all available features.
        """
        return [
            'length', 'mean', 'peak', 'std_1st_der', 'trough', 'variance',
            'spikeness', 'seasonality_strength', 'flat_spots', 'crossing_points',
            'binarize_mean', 'entropy', 'stability', 'absolute_energy', 'missing_points'
        ]
