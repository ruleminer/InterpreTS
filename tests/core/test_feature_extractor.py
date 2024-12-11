import pytest
import pandas as pd
import numpy as np
from interpreTS.core.feature_extractor import FeatureExtractor, Features


@pytest.fixture
def sample_data():
    """
    Fixture for sample time series data.
    """
    return pd.DataFrame({
        "id": [1] * 10 + [2] * 10,
        "time": pd.date_range("2023-01-01", periods=10).tolist() * 2,
        "value": np.arange(20)
    })


def test_default_feature_extraction(sample_data):
    """
    Test default feature extraction with all features enabled.
    """
    extractor = FeatureExtractor(id_column="id", sort_column="time", feature_column="value")
    features = extractor.extract_features(sample_data)
    assert not features.empty, "Features should not be empty with valid input data."


def test_custom_features(sample_data):
    """
    Test extraction with custom selected features.
    """
    extractor = FeatureExtractor(features=[Features.MEAN, Features.VARIANCE],
                                  id_column="id", sort_column="time", feature_column="value")
    features = extractor.extract_features(sample_data)
    assert not features.empty, "Features should not be empty with valid input data."
    assert all(col.startswith(("mean", "variance")) for col in features.columns if col != "id"), \
        "Only selected features should be extracted."


def test_empty_data():
    """
    Test feature extraction with empty data.
    """
    empty_data = pd.DataFrame(columns=["id", "time", "value"])
    extractor = FeatureExtractor(id_column="id", sort_column="time", feature_column="value")
    features = extractor.extract_features(empty_data)
    assert features.empty, "Features should be empty for empty input data."


def test_invalid_window_size(sample_data):
    """
    Test feature extraction with an invalid window size.
    """
    with pytest.raises(ValueError, match="Window size must be a positive number or NaN."):
        FeatureExtractor(window_size=-1, id_column="id", sort_column="time", feature_column="value")


def test_stride_effect(sample_data):
    """
    Test feature extraction with a custom stride value.
    """
    extractor = FeatureExtractor(window_size=5, stride=2, id_column="id", sort_column="time", feature_column="value")
    features = extractor.extract_features(sample_data)
    assert not features.empty, "Features should not be empty with valid input data."
    assert len(features) < len(sample_data), "Stride should reduce the number of feature rows."


def test_feature_extraction_without_sort_column(sample_data):
    """
    Test feature extraction without specifying a sort column.
    """
    extractor = FeatureExtractor(id_column="id", feature_column="value")
    features = extractor.extract_features(sample_data)
    assert not features.empty, "Features should not be empty without a sort column."
    assert "id" in features.columns, "ID column should be included in the output."


def test_feature_grouping():
    """
    Test grouping features by interpretability levels.
    """
    extractor = FeatureExtractor()
    grouped_features = extractor.group_features_by_interpretability()
    assert "easy" in grouped_features, "Interpretability groups should include 'easy'."
    assert "moderate" in grouped_features, "Interpretability groups should include 'moderate'."
    assert "advanced" in grouped_features, "Interpretability groups should include 'advanced'."


def test_generate_feature_descriptions():
    """
    Test generating feature descriptions for extracted features.
    """
    extractor = FeatureExtractor()
    extracted_features = {"mean_value": 5.0, "variance_value": 2.0}
    descriptions = extractor.generate_feature_descriptions(extracted_features)
    assert "mean_value" in descriptions, "Descriptions should include mean_value."
    assert "variance_value" in descriptions, "Descriptions should include variance_value."
    assert "Value" in descriptions["mean_value"], "Descriptions should include feature values."


def test_feature_extraction_with_nan_values():
    """
    Test feature extraction with NaN values in the data.
    """
    data_with_nan = pd.DataFrame({
        "id": [1, 1, 1],
        "time": pd.date_range("2023-01-01", periods=3),
        "value": [1, np.nan, 3]
    })
    extractor = FeatureExtractor(id_column="id", sort_column="time", feature_column="value")
    features = extractor.extract_features(data_with_nan)
    assert not features.empty, "Features should be extracted even with NaN values."


def test_feature_extraction_with_multiple_columns(sample_data):
    """
    Test feature extraction with multiple feature columns.
    """
    sample_data["value2"] = sample_data["value"] * 2
    extractor = FeatureExtractor(id_column="id", sort_column="time")
    features = extractor.extract_features(sample_data)
    assert not features.empty, "Features should be extracted for multiple columns."
    assert any("value2" in col for col in features.columns), "Features for value2 should be included."
