from .features.feature_peak import calculate_peak
from .features.feature_trough import calculate_trough
from .features.feature_length import calculate_length
from .features.feature_mean import calculate_mean
from .features.seasonality_strength import calculate_seasonality_strength
from .features.feature_variance import calculate_variance
from .features.feature_std_1st_der import calculate_std_1st_der

class FeatureExtractor:
    """
    A class to manage and execute feature extraction on time series data.
    """

    def __init__(self, features=None, feature_params=None):
        """
        Initialize the FeatureExtractor with a list of features to calculate and optional parameters for each feature.
        
        Parameters
        ----------
        features : list of str, optional
            A list of features to calculate. Default is None, which calculates all available features.
        feature_params : dict, optional
            A dictionary of parameters for specific features, where keys are feature names and values are dicts of parameters.
            For example, {'variance': {'ddof': 0}} to set ddof to 0 for the variance calculation.
        """
        
        self.features = features if features is not None else ['length']
        self.feature_params = feature_params if feature_params is not None else {}

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
            peak_params = self.feature_params.get('peak', {})
            extracted_features['peak'] = calculate_peak(data, **peak_params)

        if 'std_1st_der' in self.features:
            extracted_features['std_1st_der'] = calculate_std_1st_der(data)

        if 'trough' in self.features:
            trough_params = self.feature_params.get('trough', {})
            extracted_features['trough'] = calculate_trough(data, **trough_params)

        if 'variance' in self.features:
            variance_params = self.feature_params.get('variance', {})
            extracted_features['variance'] = calculate_variance(data, **variance_params)

        if 'spikeness' in self.features:
            extracted_features['spikeness'] = calculate_spikeness(data)  

        if 'seasonality_strength' in self.features:
            extracted_features['seasonality_strength'] = calculate_seasonality_strength(data, frequency=7)

        return extracted_features
