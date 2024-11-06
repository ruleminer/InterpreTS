import pandas as pd
import numpy as np
from src.core.feature_extractor import FeatureExtractor, Features

def test_extract_length_feature():
    """
    Test that FeatureExtractor correctly extracts the 'length' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=[Features.LENGTH])
    features = extractor.extract_features(data)
    assert features[Features.LENGTH] == 5, "The 'length' feature should be 5"

def test_extract_mean_feature():
    """
    Test that FeatureExtractor correctly extracts the 'mean' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=[Features.MEAN])
    features = extractor.extract_features(data)
    assert features[Features.MEAN] == 3.0, "The 'mean' feature should be 3.0"

def test_extract_spikeness_feature():
    """
    Test that FeatureExtractor correctly extracts the 'spikeness' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
    extractor = FeatureExtractor(features=[Features.SPIKENESS])
    features = extractor.extract_features(data)
    assert np.isclose(features[Features.SPIKENESS], 0, atol=0.2), "The 'spikeness' feature should be close to 0 for symmetric data"

def test_extract_std_1st_der_feature():
    """
    Test that FeatureExtractor correctly extracts the 'std_1st_der' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=[Features.STD_1ST_DER])
    features = extractor.extract_features(data)
    assert np.isclose(features[Features.STD_1ST_DER], 0, atol=1e-1), "The 'std_1st_der' feature should be close to 0 for linear increase"

def test_extract_seasonality_strength_feature():
    """
    Test that FeatureExtractor correctly extracts the 'seasonality_strength' feature.
    """
    data = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    extractor = FeatureExtractor(features=[Features.CALCULATE_SEASONALITY_STRENGTH],
                                 feature_params={Features.CALCULATE_SEASONALITY_STRENGTH: {'period': 2}})
    features = extractor.extract_features(data)
    assert features[Features.CALCULATE_SEASONALITY_STRENGTH] > 0.5, "The 'seasonality_strength' feature should be high for repeating patterns"
