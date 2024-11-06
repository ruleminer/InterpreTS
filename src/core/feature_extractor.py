from .features.feature_spikeness import calculate_spikeness
from .features.feature_peak import calculate_peak
from .features.feature_trough import calculate_trough
from .features.feature_length import calculate_length
from .features.feature_mean import calculate_mean
from .features.seasonality_strength import calculate_seasonality_strength
from .features.feature_variance import calculate_variance
from .features.feature_std_1st_der import calculate_std_1st_der

class Features:
    LENGTH = 'length'
    MEAN = 'mean'
    PEAK = 'peak'
    STD_1ST_DER = 'std_1st_der'
    TROUGH = 'trough'
    VARIANCE = 'variance'
    SPIKENESS = 'spikeness'
    CALCULATE_SEASONALITY_STRENGTH = 'seasonality_strength'

class FeatureExtractor:
    """
    A class to manage and execute feature extraction on time series data.
    """

    def __init__(self, features=None, feature_params=None):
        """
        Initialize the FeatureExtractor with a list of features to calculate and optional parameters for each feature.
        
        Parameters
        ----------
        features : list of Features constants, optional
            A list of features to calculate. Default is None, which calculates all available features.
        feature_params : dict, optional
            A dictionary of parameters for specific features, where keys are feature names and values are dicts of parameters.
            For example, {'variance': {'ddof': 0}} to set ddof to 0 for the variance calculation.
        """
        
        self.features = features if features is not None else [Features.LENGTH]
        self.feature_params = feature_params if feature_params is not None else {}

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
        }

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
        """

        extracted_features = {}

        # Iterate through selected features and calculate them
        for feature_name in self.features:
            if feature_name in self.feature_functions:
                # Retrieve any parameters specific to this feature
                params = self.feature_params.get(feature_name, {})
                
                # Call the feature calculation function with parameters
                extracted_features[feature_name] = self.feature_functions[feature_name](data, **params)

        return extracted_features

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
