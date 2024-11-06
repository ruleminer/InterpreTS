import pandas as pd
import numpy as np
from src.core.feature_extractor import FeatureExtractor

def test_extract_peak_feature_with_params():
    """
    Test that FeatureExtractor correctly extracts the 'peak' feature with specified parameters.
    """
    data = pd.Series([1, 2, 3, 4, 5, 3, 2, 1])
    extractor = FeatureExtractor(features=['peak'], feature_params={'peak': {'start': 1, 'end': 5}})
    features = extractor.extract_features(data)
    assert features['peak'] == 5.0, "The 'peak' feature should be 5.0 within the specified range"

def test_extract_trough_feature_with_params():
    """
    Test that FeatureExtractor correctly extracts the 'trough' feature with specified parameters.
    """
    data = pd.Series([5, 4, 3, 2, 1, 2, 3, 4])
    extractor = FeatureExtractor(features=['trough'], feature_params={'trrough': {'start': 1, 'end': 5}})
    features = extractor.extract_features(data)
    assert features['trough'] == 1.0, "The 'trough' feature should be 1.0 within the specified range"

def test_extract_variance_feature_with_ddof():
    """
    Test that FeatureExtractor correctly extracts the 'variance' feature with specified ddof.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=['variance'], feature_params={'variance': {'ddof': 0}})
    features = extractor.extract_features(data)
    assert features['variance'] == 2.0, "The 'variance' feature should be 2.0 with ddof set to 0"

def test_extract_multiple_features_with_params():
    """
    Test that FeatureExtractor can extract 'peak', 'trough', and 'variance' features with parameters.
    """
    data = pd.Series([1, 2, 3, 2, 1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=['peak', 'trough', 'variance'],
                                 feature_params={'peak': {'start': 2, 'end': 8},
                                                 'trough': {'start': 2, 'end': 7},
                                                 'variance': {'ddof': 0}})
    features = extractor.extract_features(data)
    assert features['peak'] == 4.0, "The 'peak' feature should be 4.0 within the specified range"
    assert features['trough'] == 1.0, "The 'trough' feature should be 1.0 within the specified range"
    assert np.isclose(features['variance'], 1.5802, atol=1e-4), "The 'variance' feature should be 1.5802 with ddof set to 0"
