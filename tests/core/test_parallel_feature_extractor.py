import pytest
import pandas as pd
import numpy as np
from interpreTS.core.parallel_feature_extractor import ParallelFeatureExtractor

def test_sequential_extraction():
    """
    Test feature extraction in sequential mode.
    """
    data = pd.Series(range(10))
    extractor = ParallelFeatureExtractor(features=['mean', 'variance'], window_size=5, stride=1)
    result = extractor.extract_features(data)
    
    assert len(result) == 6, "The number of windows should be 6 for window_size=5 and stride=1"
    assert 'mean' in result.columns, "Mean feature should be calculated"
    assert 'variance' in result.columns, "Variance feature should be calculated"
    assert np.isclose(result['mean'].iloc[0], 2, atol=1e-4), "Mean for the first window should be 2"
    assert np.isclose(result['variance'].iloc[0], 2.5, atol=1e-4), "Variance for the first window should be 2.5"


@pytest.mark.parametrize("library", ["dask", "ray"])
def test_parallel_extraction(library):
    """
    Test feature extraction in parallel mode using Dask or Ray.
    """
    data = pd.Series(range(20))
    use_dask = library == "dask"
    use_ray = library == "ray"

    extractor = ParallelFeatureExtractor(features=['mean', 'variance'], window_size=5, stride=2, use_dask=use_dask, use_ray=use_ray)
    result = extractor.extract_features(data)

    assert len(result) == 8, "The number of windows should be 8 for window_size=5 and stride=2"
    assert 'mean' in result.columns, "Mean feature should be calculated"
    assert 'variance' in result.columns, "Variance feature should be calculated"
    assert np.isclose(result['mean'].iloc[0], 2, atol=1e-4), "Mean for the first window should be 2"
    assert np.isclose(result['variance'].iloc[0], 2.5, atol=1e-4), "Variance for the first window should be 2.5"


def test_empty_data():
    """
    Test behavior with empty input data.
    """
    data = pd.Series([])
    extractor = ParallelFeatureExtractor(features=['mean', 'variance'], window_size=5, stride=1)
    result = extractor.extract_features(data)

    assert result.empty, "Result should be empty for empty input data"


def test_insufficient_window_size():
    """
    Test behavior when the window size is larger than the data length.
    """
    data = pd.Series([1, 2, 3])
    extractor = ParallelFeatureExtractor(features=['mean', 'variance'], window_size=5, stride=1)
    result = extractor.extract_features(data)

    assert result.empty, "Result should be empty when window size is larger than data length"


def test_multiple_features():
    """
    Test extraction of multiple features in parallel mode.
    """
    data = pd.Series(range(10))
    extractor = ParallelFeatureExtractor(features=['mean', 'variance', 'spikeness'], window_size=5, stride=1, use_dask=True)
    result = extractor.extract_features(data)

    assert 'mean' in result.columns, "Mean feature should be calculated"
    assert 'variance' in result.columns, "Variance feature should be calculated"
    assert 'spikeness' in result.columns, "Spikeness feature should be calculated"
    assert len(result) == 6, "The number of windows should be 6 for window_size=5 and stride=1"


def test_ray_parallel_feature_extraction():
    """
    Test feature extraction with Ray in parallel mode.
    """
    data = pd.Series(range(30))
    extractor = ParallelFeatureExtractor(features=['mean', 'variance'], window_size=10, stride=5, use_ray=True)
    result = extractor.extract_features(data)

    assert len(result) == 5, "The number of windows should be 5 for window_size=10 and stride=5"
    assert 'mean' in result.columns, "Mean feature should be calculated"
    assert 'variance' in result.columns, "Variance feature should be calculated"
    assert np.isclose(result['mean'].iloc[0], 4.5, atol=1e-4), "Mean for the first window should be 4.5"


def test_feature_parameters():
    """
    Test feature extraction with parameters for specific features.
    """
    data = pd.Series(range(10))
    extractor = ParallelFeatureExtractor(
        features=['seasonality_strength'],
        feature_params={'seasonality_strength': {'period': 2}},
        window_size=5,
        stride=1,
        use_dask=True
    )
    result = extractor.extract_features(data)

    assert 'seasonality_strength' in result.columns, "Seasonality strength feature should be calculated"
    assert len(result) == 6, "The number of windows should be 6 for window_size=5 and stride=1"
