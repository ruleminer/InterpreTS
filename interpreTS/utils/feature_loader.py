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
    
class FeatureLoader:
    
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

        feature_constants = {
            name: value
            for name, value in Features.__dict__.items()
            if not name.startswith('__') and not callable(value)
        }

        return {name.capitalize(): constant for name, constant in feature_constants.items()}
