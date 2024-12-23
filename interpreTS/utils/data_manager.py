from .feature_loader import Features
from ..core.features.feature_spikeness import calculate_spikeness
from ..core.features.feature_entropy import calculate_entropy
from ..core.features.feature_stability import calculate_stability
from ..core.features.feature_length import calculate_length
from ..core.features.feature_mean import calculate_mean
from ..core.features.seasonality_strength import calculate_seasonality_strength
from ..core.features.feature_variance import calculate_variance
from ..core.features.feature_peak import calculate_peak
from ..core.features.feature_trough import calculate_trough
from ..core.features.feature_heterogeneity import calculate_heterogeneity
from ..core.features.feature_absolute_energy import calculate_absolute_energy
from ..core.features.feature_missing_points import calculate_missing_points
from ..core.features.distance_to_the_last_change_point import calculate_distance_to_last_trend_change
from ..core.features.feature_above_9th_decile import calculate_above_9th_decile
from ..core.features.feature_below_1st_decile import calculate_below_1st_decile
from ..core.features.feature_binarize_mean import calculate_binarize_mean
from ..core.features.feature_crossing_points import calculate_crossing_points
from ..core.features.feature_flat_spots import calculate_flat_spots
from ..core.features.feature_outliers_iqr import calculate_outliers_iqr
from ..core.features.feature_outliers_std import calculate_outliers_std
from ..core.features.feature_std_1st_der import calculate_std_1st_der
from ..core.features.histogram_dominant import calculate_dominant
from ..core.features.mean_change import calculate_mean_change
from ..core.features.trend_strength import calculate_trend_strength
from ..core.features.feature_significant_changes import calculate_significant_changes
from ..core.features.variability_in_sub_periods import calculate_variability_in_sub_periods
from ..core.features.variance_change import calculate_change_in_variance
from ..core.features.feature_linearity import calculate_linearity

def load_metadata():
    return {
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
        },
        Features.STD_1ST_DER: {
            'level': 'moderate',
            'description': 'Standard deviation of the first derivative of the signal.'
        },
        Features.DOMINANT: {
            'level': 'moderate',
            'description': 'The dominant value of the time series histogram, representing the most frequent range of values within the specified bins.'
        },
        Features.MEAN_CHANGE: {
            'level': 'moderate',
            'description': 'The rate of change in the rolling mean over time, capturing trends or shifts in the time series.'
        },
        Features.TREND_STRENGTH: {
            'level': 'moderate',
            'description': 'The R-squared value from a linear regression, representing the strength and consistency of the trend in the time series.'
        },
        Features.SIGNIFICANT_CHANGES: {
            'level': 'moderate',
            'description': 'Proportion of significant increases or decreases in the time series, based on deviations from the interquartile range (IQR) of differences between consecutive values.'
        },
        Features.MISSING_POINTS: {
            'level': 'easy',
            'description': 'Proportion or count of missing data points in the window.'
        },
        Features.VARIABILITY_IN_SUB_PERIODS: {
            'level': 'moderate',
            'description': 'Variance calculated within sub-periods of a time series, providing a measure of variability across fixed-size windows.'
        },
        Features.CHANGE_IN_VARIANCE: {
            'level': 'moderate',
            'description': 'Change in variance over time, calculated as the difference between rolling variances across consecutive windows.'
        },
        Features.LINEARITY:{
            'level': 'moderate',
            'description': 'Measure of how well the time series can be approximated by a linear trend, quantified using the R-squared value from linear regression.'
        }
    } 

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
    feature_metadata = self.load_metadata()
    for feature_name, feature_value in extracted_features.items():
        if feature_name in feature_metadata:
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

def load_feature_functions():
    return {
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
            Features.HETEROGENEITY: calculate_heterogeneity,
            Features.ABSOLUTE_ENERGY: calculate_absolute_energy,
            Features.MISSING_POINTS: calculate_missing_points,
            Features.ABOVE_9TH_DECILE: calculate_above_9th_decile,
            Features.BELOW_1ST_DECILE: calculate_below_1st_decile,
            Features.BINARIZE_MEAN: calculate_binarize_mean,
            Features.CROSSING_POINTS: calculate_crossing_points,
            Features.FLAT_SPOTS: calculate_flat_spots,
            Features.OUTLIERS_IQR: calculate_outliers_iqr,
            Features.OUTLIERS_STD: calculate_outliers_std,
            Features.STD_1ST_DER: calculate_std_1st_der,
            Features.DOMINANT: calculate_dominant,
            Features.MEAN_CHANGE: calculate_mean_change,
            Features.TREND_STRENGTH: calculate_trend_strength,
            Features.SIGNIFICANT_CHANGES: calculate_significant_changes,
            Features.VARIABILITY_IN_SUB_PERIODS: calculate_variability_in_sub_periods,
            Features.CHANGE_IN_VARIANCE: calculate_change_in_variance,
            Features.LINEARITY: calculate_linearity
        }
    
def load_validation_requirements():
    return {
            Features.LINEARITY: {
                "require_datetime_index": False,
                "allow_nan": True,  
                "check_one_dimensional": True,  
            },
            Features.MEAN: {
                "require_datetime_index": False,
                "allow_nan": False, 
                "check_one_dimensional": True,
            },
            Features.VARIANCE: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True
            },
            Features.SPIKENESS: {
                "require_datetime_index": False,
                "allow_nan": True,
                "check_one_dimensional": True
            },
            Features.ENTROPY: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 2
            },
            Features.STABILITY: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 2
            },
            Features.SEASONALITY_STRENGTH: {
                "require_datetime_index": False,
                "allow_nan": True,
                "check_one_dimensional": True,
                "min_length": 2,
                "validate_positive_parameters": {"period": "Period must be a positive integer."}
            },
            Features.PEAK: {
                "require_datetime_index": False,
                "allow_nan": True,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.TROUGH: {
                "require_datetime_index": False,
                "allow_nan": True,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.DISTANCE_TO_LAST_TREND_CHANGE: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": "window_size + 1"
            },
            Features.HETEROGENEITY: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1,
                "check_nonzero_mean": True
            },
            Features.ABSOLUTE_ENERGY: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.MISSING_POINTS: {
                "require_datetime_index": False,
                "allow_nan": True,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.ABOVE_9TH_DECILE: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1,
                "additional_requirements": {
                    "training_data": {
                        "require_datetime_index": False,
                        "allow_nan": False,
                        "check_one_dimensional": True,
                        "min_length": 1
                    }
                }
            },
            Features.BELOW_1ST_DECILE: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1,
                "additional_requirements": {
                    "training_data": {
                        "require_datetime_index": False,
                        "allow_nan": False,
                        "check_one_dimensional": True,
                        "min_length": 1
                    }
                }
            },
            Features.BINARIZE_MEAN: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.CROSSING_POINTS: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.FLAT_SPOTS: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.OUTLIERS_IQR: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1,
                "additional_requirements": {
                    "training_data_not_empty": True,
                    "training_data_no_nan": True
                }
            },
            Features.OUTLIERS_STD: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1,
                "additional_requirements": {
                    "training_data_not_empty": True,
                    "training_data_no_nan": True
                }
            },
            Features.STD_1ST_DER: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.DOMINANT: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 1
            },
            Features.MEAN_CHANGE: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 2,
                "positive_integer_parameters": ["window_size"]
            },
            Features.TREND_STRENGTH: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 2
            },
            Features.SIGNIFICANT_CHANGES: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 2
            },
            Features.VARIABILITY_IN_SUB_PERIODS: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": "window_size"
            },
            Features.CHANGE_IN_VARIANCE: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": "window_size",
                "positive_integer_params": ["window_size"]
            },
            Features.LINEARITY: {
                "require_datetime_index": False,
                "allow_nan": False,
                "check_one_dimensional": True,
                "min_length": 2
            }
        }