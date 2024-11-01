from .features.feature_length import calculate_feature_length

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
            extracted_features['length'] = calculate_feature_length(data)
                
        return extracted_features
