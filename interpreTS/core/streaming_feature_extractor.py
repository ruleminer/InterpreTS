import numpy as np
import pandas as pd
from collections import deque

class StreamingFeatureExtractor:
    def __init__(self, features, window_size=5, stride=1, feature_params=None):
        self.features = features  # List of feature names to compute
        self.window_size = window_size
        self.stride = stride
        self.feature_params = feature_params or {}
        self.buffer = deque(maxlen=window_size)
        self.current_position = 0

    def add_data(self, new_data):
        """
        Add new data points to the stream and calculate features if a full window is available.
        
        Parameters
        ----------
        new_data : list, np.array, or pd.Series
            New incoming data points.
        
        Returns
        -------
        List[dict]
            A list of feature dictionaries for each complete window.
        """
        results = []

        # Extend buffer with new data points
        for point in new_data:
            self.buffer.append(point)
            if len(self.buffer) == self.window_size:
                # Calculate features on the current window
                window_data = pd.Series(list(self.buffer))
                features = self.extract_features(window_data)
                results.append(features)

                # Move the window forward by stride length
                for _ in range(self.stride):
                    if self.buffer:
                        self.buffer.popleft()
        return results

    def extract_features(self, data):
        """
        Extract the desired features from the current data window.
        
        Parameters
        ----------
        data : pd.Series
            The time series data in the current window.
        
        Returns
        -------
        dict
            Dictionary of computed features.
        """
        feature_results = {}
        for feature in self.features:
            # Retrieve and apply each feature calculation function
            feature_func = self.feature_params.get(feature)
            if feature_func:
                feature_results[feature] = feature_func(data)
        return feature_results
