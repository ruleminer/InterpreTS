import pytest
import pandas as pd
from src.core.feature_extractor import FeatureExtractor

def test_extract_length_feature():
    """
    Test that FeatureExtractor correctly extracts the 'length' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=['length'])
    features = extractor.extract_features(data)
    assert features['length'] == 5, "The 'length' feature should be 5"

def test_extract_no_features():
    """
    Test that FeatureExtractor returns an empty dictionary when no features are specified.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=[])
    features = extractor.extract_features(data)
    assert features == {}, "No features should be extracted when an empty feature list is provided"
