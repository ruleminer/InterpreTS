from .features.feature_length import calculate_length
from .features.feature_mean import calculate_mean
from .features.feature_peak import calculate_peak
from .features.feature_trough import calculate_trough
from .features.seasonality_strength import calculate_seasonality_strength

class FeatureExtractor:
    """
    A class to manage and execute feature extraction on time series data.
    """

    def __init__(self, features=None):
        """
        Initialize the FeatureExtractor with a list of features to calculate.
        
        Parameters
        ----------
        features : list of str, optional
            A list of features to calculate. Default is None, which calculates all available features.
        """
        
        
        self.features = features if features is not None else ['length']

    def extract_features(self, data):
        """
        Extract features from a time series dataset.
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            The time series data for which features are to be extracted.
            
        Returns
        -------
        dict
            A dictionary containing calculated features and their values.
            
        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> extractor = FeatureExtractor(features=['length'])
        >>> extractor.extract_features(data)
        {'length': 5}
        """


        extracted_features = {}

        if 'length' in self.features:
            extracted_features['length'] = calculate_length(data)

        if 'mean' in self.features:
            extracted_features['mean'] = calculate_mean(data)

        if 'peak' in self.features:
            extracted_features['peak'] = calculate_peak(data)

        if 'trough' in self.features:
            extracted_features['trough'] = calculate_trough(data)

        if 'seasonality_strength' in self.features:
            extracted_features['seasonality_strength'] = calculate_seasonality_strength(data, frequency=7)

        return extracted_features
