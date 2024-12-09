import pandas as pd
import numpy as np
from interpreTS.core.feature_extractor import FeatureExtractor, Features
import pytest

def test_feature_extractor_initialization():
    extractor = FeatureExtractor()
    assert np.isnan(extractor.window_size), "Default window_size should be NaN."
    assert extractor.stride == 1, "Default stride should be 1."
    assert extractor.features == extractor.DEFAULT_FEATURES, "Default features should include predefined default features."
    assert extractor.feature_column == "value", "Default feature_column should be 'value'."

def test_feature_extractor_full_length_window():
    data = pd.Series([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(features=[Features.LENGTH], window_size=np.nan)
    features = extractor.extract_features(data)
    assert len(features) == 1, "There should be a single feature set for the full-length window."
    assert Features.LENGTH in features.columns, "'length' should be a calculated feature."
    assert features[Features.LENGTH].iloc[0] == 5, "The 'length' feature should equal the length of the series."

def test_feature_extractor_with_window_and_stride():
    data = pd.Series(range(10), name='value')
    extractor = FeatureExtractor(features=[Features.LENGTH], window_size=3, stride=2)
    features = extractor.extract_features(data)
    expected_num_windows = (len(data) - extractor.window_size) // extractor.stride + 1
    assert len(features) == expected_num_windows, (
        f"Number of windows should match (len(data) - window_size) // stride + 1. "
        f"Expected {expected_num_windows}, got {len(features)}"
    )
    assert Features.LENGTH in features.columns, f"Expected {Features.LENGTH} in feature columns."
    assert all(features[Features.LENGTH] == 3), "All windows should have a 'length' feature equal to the window size."

def test_feature_extractor_custom_feature_column():
    data = pd.DataFrame({
        'time': [1, 2, 3, 4, 5],
        'custom_value': [10, 20, 30, 40, 50]
    })
    extractor = FeatureExtractor(features=[Features.MEAN], feature_column='custom_value', window_size=np.nan)
    features = extractor.extract_features(data)
    assert len(features) == 1, "There should be a single feature set for the full-length window."
    assert Features.MEAN in features.columns, "'mean' should be a calculated feature."
    assert np.isclose(features[Features.MEAN].iloc[0], 30, atol=1e-4), "The mean should be correctly calculated."

def test_feature_extractor_with_id_column():
    data = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'value': [1, 2, 3, 4, 5, 6]
    })
    extractor = FeatureExtractor(features=[Features.MEAN], id_column='id', window_size=3)
    features = extractor.extract_features(data)
    assert len(features) == 2, "Each ID should have its own group of extracted features."
    assert features['id'].nunique() == 2, "Extracted features should have two unique IDs."
    assert np.isclose(features[Features.MEAN].iloc[0], 2, atol=1e-4), "Mean for ID 1 should be calculated correctly."
    assert np.isclose(features[Features.MEAN].iloc[1], 5, atol=1e-4), "Mean for ID 2 should be calculated correctly."
def test_feature_extractor_sort_column():
    data = pd.DataFrame({
        'time': [3, 1, 2],
        'value': [10, 20, 30]
    })
    extractor = FeatureExtractor(features=[Features.MEAN], sort_column='time', window_size=3)
    features = extractor.extract_features(data)
    assert features.shape[0] == 1, "After sorting and windowing, only one window should remain."
    assert np.isclose(features[Features.MEAN].iloc[0], 20, atol=1e-4), "The mean after sorting by 'time' should be calculated correctly."

def test_feature_extractor_no_sorting_or_grouping():
    data = pd.Series(range(10))
    extractor = FeatureExtractor(window_size=4, stride=2)
    features = extractor.extract_features(data)
    expected_num_windows = (len(data) - extractor.window_size) // extractor.stride + 1
    assert len(features) == expected_num_windows, "Number of windows should be calculated based on window_size and stride."
    assert Features.LENGTH in features.columns, "Expected 'length' in extracted features."
    assert all(features[Features.LENGTH] == 4), "Each extracted window should have a 'length' of 4."

def test_feature_extractor_empty_data():
    data = pd.Series([], dtype=float)
    extractor = FeatureExtractor(features=[Features.LENGTH])
    features = extractor.extract_features(data)
    assert features.empty, "Features should be empty for empty input data."

def test_invalid_window_size():
    with pytest.raises(ValueError, match="Window size must be a positive number or NaN."):
        FeatureExtractor(window_size=-1)

def test_invalid_feature_column():
    with pytest.raises(ValueError, match="Feature column must be a string."):
        FeatureExtractor(feature_column=123)

def test_invalid_stride():
    with pytest.raises(ValueError, match="Stride must be a positive integer."):
        FeatureExtractor(stride=0)

def test_valid_initialization():
    extractor = FeatureExtractor(
        features=[Features.LENGTH, Features.MEAN],
        feature_params={Features.MEAN: {"param1": 10}},
        window_size=5,
        stride=2,
        id_column="id",
        sort_column="time",
        feature_column="value"
    )
    assert extractor.features == [Features.LENGTH, Features.MEAN], "Features should match the input list."
    assert extractor.feature_params == {Features.MEAN: {"param1": 10}}, "Feature params should match the input dictionary."
    assert extractor.window_size == 5, "Window size should match the input value."
    assert extractor.stride == 2, "Stride should match the input value."

def test_extract_features_with_nan_window_size():
    data = pd.Series(range(10))
    extractor = FeatureExtractor(window_size=np.nan)
    features = extractor.extract_features(data)
    assert len(features) == 1, "With window_size NaN, the entire series should be a single window."
    assert features[Features.LENGTH].iloc[0] == len(data), "Length feature should equal the series length."