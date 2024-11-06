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
    length_value = features[Features.LENGTH].iloc[0] if isinstance(features[Features.LENGTH], pd.Series) else features[Features.LENGTH]
    assert length_value == 5, "The 'length' feature should be 5"
    
def test_extract_mean_feature():
    """
    Test that FeatureExtractor correctly extracts the 'mean' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=[Features.MEAN])
    features = extractor.extract_features(data)
    assert np.isclose(features[Features.MEAN], 3.0, atol=1e-4), "The 'mean' feature should be 3.0"

def test_extract_spikeness_feature():
    """
    Test that FeatureExtractor correctly extracts the 'spikeness' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
    extractor = FeatureExtractor(features=[Features.SPIKENESS])
    features = extractor.extract_features(data)
    spikeness_value = features[Features.SPIKENESS].iloc[0] if isinstance(features[Features.SPIKENESS], pd.Series) else features[Features.SPIKENESS]
    assert np.isclose(spikeness_value, 0, atol=0.2), "The 'spikeness' feature should be close to 0 for symmetric data"

def test_extract_std_1st_der_feature():
    """
    Test that FeatureExtractor correctly extracts the 'std_1st_der' feature.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=[Features.STD_1ST_DER])
    features = extractor.extract_features(data)
    assert np.isclose(features[Features.STD_1ST_DER], 0, atol=1e-2), "The 'std_1st_der' feature should be approximately 0 for a linearly increasing sequence"

def test_extract_seasonality_strength_feature():
    """
    Test that FeatureExtractor correctly extracts the 'seasonality_strength' feature.
    """
    data = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    extractor = FeatureExtractor(features=[Features.CALCULATE_SEASONALITY_STRENGTH],
                                 feature_params={Features.CALCULATE_SEASONALITY_STRENGTH: {'period': 2}})
    features = extractor.extract_features(data)
    
    assert features[Features.CALCULATE_SEASONALITY_STRENGTH].iloc[0] > 0.5, "The 'seasonality_strength' feature should be high for repeating patterns"

def test_extract_with_window_size_and_stride():
    """
    Test that FeatureExtractor correctly applies window_size and stride.
    """
    data = pd.DataFrame({'value': range(10)})
    extractor = FeatureExtractor(features=[Features.MEAN], window_size=3, stride=2)
    features = extractor.extract_features(data)    
    expected_means = [1, 3, 5, 7]
    actual_means = features[Features.MEAN].tolist()
    assert actual_means == expected_means, f"Expected means {expected_means}, got {actual_means}"


def test_extract_with_id_column():
    """
    Test that FeatureExtractor groups data by id_column before feature extraction.
    """
    data = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'value': [1, 2, 3, 4, 5, 6]
    })
    extractor = FeatureExtractor(features=[Features.MEAN], id_column='id', window_size=3)
    features = extractor.extract_features(data)
    
    assert len(features) == 2, "Powinny być 2 grupy dla id_column 'id'"
    assert features['id'].nunique() == 2, "Każda grupa powinna mieć unikalne id"

def test_extract_variance_feature():
    """
    Test that FeatureExtractor correctly extracts the 'variance' feature.
    """
    data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    extractor = FeatureExtractor(features=[Features.VARIANCE])
    features = extractor.extract_features(data)
    assert np.isclose(features[Features.VARIANCE].iloc[0], 2.0, atol=1e-4), "The 'variance' feature should be 2.0"

def test_extract_peak_and_trough_features():
    """
    Test that FeatureExtractor correctly extracts the 'peak' and 'trough' features.
    """
    data = pd.DataFrame({'value': [1, 3, 2, 4, 1]})
    extractor = FeatureExtractor(features=[Features.PEAK, Features.TROUGH])
    features = extractor.extract_features(data)
    
    assert features[Features.PEAK].iloc[0] == 4, "The peak feature should be 4"
    assert features[Features.TROUGH].iloc[0] == 1, "The trough feature should be 1"

def test_extract_with_sort_column():
    """
    Test that FeatureExtractor sorts data by sort_column before feature extraction.
    """
    data = pd.DataFrame({
        'id': [1, 1, 1],
        'time': [3, 1, 2],
        'value': [10, 20, 30]
    })
    extractor = FeatureExtractor(features=[Features.MEAN], sort_column='time', window_size=3, stride=1)
    features = extractor.extract_features(data)
    assert Features.MEAN in features.columns, "The 'mean' feature should be in the features DataFrame"
    assert features[Features.MEAN].iloc[0] == 20, "The mean should be calculated after sorting by 'time'"
