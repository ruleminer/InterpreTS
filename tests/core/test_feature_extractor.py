import pytest
import pandas as pd
import numpy as np
from interpreTS.core.feature_extractor import FeatureExtractor, Features

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
    assert np.isclose(features[Features.VARIANCE].iloc[0], 2.5, atol=1e-4), "The 'variance' feature should be 2.5"

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

def test_extract_absolute_energy_feature():
    """
    Test that FeatureExtractor correctly extracts the 'absolute_energy' feature.
    """
    data = pd.Series([1, 2, 3, -4, 5])
    extractor = FeatureExtractor(features=[Features.ABSOLUTE_ENERGY])
    features = extractor.extract_features(data)
    assert np.isclose(features[Features.ABSOLUTE_ENERGY].iloc[0], 55.0, atol=1e-4), "The 'absolute_energy' feature should be 55.0"
    
def test_extract_missing_points_feature():
    """
    Test that FeatureExtractor correctly extracts the 'missing_points' feature.
    """
    data = pd.DataFrame({'value': [1, 3, None, 1, np.nan]})
    extractor = FeatureExtractor(features=[Features.MISSING_POINTS])
    features = extractor.extract_features(data)
    extractor2 = FeatureExtractor(features=[Features.MISSING_POINTS, ], feature_params={Features.MISSING_POINTS: {'percentage': False}})
    features2 = extractor2.extract_features(data)
    
    assert features[Features.MISSING_POINTS].iloc[0] == 0.4, "The percentage of missing points should be 40%"
    assert features2[Features.MISSING_POINTS].iloc[0] == 2, "The amount of missing points should be 2"

def test_extract_entropy_feature():
    """
    Test that FeatureExtractor correctly extracts the 'entropy' feature.
    """
    data = pd.Series([1, -2, 9, -10, -15])
    extractor = FeatureExtractor(features=[Features.ENTROPY])
    features = extractor.extract_features(data)
    assert 0 <= features[Features.ENTROPY].iloc[0] <= 1, "The 'entropy' feature should be between 0 and 1"

def test_extract_stability_features():
    """
    Test that FeatureExtractor correctly extracts the 'peak' and 'trough' features.
    """
    data = pd.DataFrame({'value': [1, 3, 1, 3, 1]})
    data2 = pd.DataFrame({'value': [1, 1, 1, 1, 1]})
    extractor = FeatureExtractor(features=[Features.STABILITY])
    features = extractor.extract_features(data)
    features2 = extractor.extract_features(data2)
    assert features[Features.STABILITY].iloc[0] < 1, "The stability feature should be less than 1 if the data is not constant"
    assert features2[Features.STABILITY].iloc[0] == 1, "The stability feature should be 1 for constant data"

def test_group_features_by_interpretability():
    """
    Test that features are correctly grouped by interpretability levels.
    """
    extractor = FeatureExtractor(features=None)  # Test all features
    groups = extractor.group_features_by_interpretability()

    assert 'easy' in groups and 'moderate' in groups and 'advanced' in groups, \
        "Groups should include 'easy', 'moderate', and 'advanced'."

    assert Features.LENGTH in groups['easy'], "'length' should be in the 'easy' group."
    assert Features.VARIANCE in groups['moderate'], "'variance' should be in the 'moderate' group."
    assert Features.ENTROPY in groups['advanced'], "'entropy' should be in the 'advanced' group."


def test_generate_feature_descriptions():
    """
    Test that textual descriptions are correctly generated for features.
    """
    extractor = FeatureExtractor(features=[Features.MEAN, Features.ENTROPY])
    extracted_features = {
        Features.MEAN: 10,
        Features.ENTROPY: 0.85
    }
    descriptions = extractor.generate_feature_descriptions(extracted_features)

    assert Features.MEAN in descriptions, "'mean' should have a description."
    assert "Mean value within the window." in descriptions[Features.MEAN], \
        "Description for 'mean' should mention 'mean value within the window'."

    assert Features.ENTROPY in descriptions, "'entropy' should have a description."
    assert "Degree of randomness or disorder" in descriptions[Features.ENTROPY], \
        "Description for 'entropy' should mention 'degree of randomness or disorder'."

def test_group_and_descriptions_integration():
    """
    Test that feature grouping and descriptions work together seamlessly.
    """
    extractor = FeatureExtractor(features=[Features.MEAN, Features.ENTROPY, Features.LENGTH])
    
    # Group features
    groups = extractor.group_features_by_interpretability()
    assert len(groups['easy']) > 0, "There should be at least one 'easy' feature."
    assert len(groups['advanced']) > 0, "There should be at least one 'advanced' feature."

    # Extract features and generate descriptions
    extracted_features = {
        Features.MEAN: 12.5,
        Features.ENTROPY: 0.92,
        Features.LENGTH: 50
    }
    descriptions = extractor.generate_feature_descriptions(extracted_features)
    for feature, desc in descriptions.items():
        assert feature in extracted_features, f"Description should exist for feature: {feature}"


def test_generate_all_feature_descriptions():
    """
    Test that descriptions are generated for all defined features.
    """
    extractor = FeatureExtractor(features=FeatureExtractor.available_features())
    extracted_features = {feature: idx + 1 for idx, feature in enumerate(FeatureExtractor.available_features())}
    
    descriptions = extractor.generate_feature_descriptions(extracted_features)
    assert len(descriptions) == len(extracted_features), "Descriptions should be generated for all features."

    for feature, desc in descriptions.items():
        assert feature in extractor.feature_metadata or "Unknown feature" in desc, \
            f"Every feature should have a corresponding description or marked as unknown."
