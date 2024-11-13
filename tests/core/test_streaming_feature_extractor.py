import pytest
import pandas as pd
from interpreTS.core.streaming_feature_extractor import StreamingFeatureExtractor

def test_streaming_extractor_with_mean():
    """
    Test that StreamingFeatureExtractor calculates mean correctly.
    """
    extractor = StreamingFeatureExtractor(features=['mean'], window_size=3)
    data_points = [1, 2, 3]
    for point in data_points:
        result = extractor.add_data(point)
    assert result['mean'] == 2.0, "Mean should be 2.0 for [1, 2, 3]"

def test_streaming_extractor_with_variance():
    """
    Test that StreamingFeatureExtractor calculates variance correctly.
    """
    extractor = StreamingFeatureExtractor(features=['variance'], window_size=3)
    data_points = [1, 2, 3]
    for point in data_points:
        result = extractor.add_data(point)
    assert result['variance'] == 1.0, "Variance should be 1.0 for [1, 2, 3]"

def test_streaming_extractor_with_crossing_points():
    """
    Test that StreamingFeatureExtractor calculates crossing points correctly.
    """
    extractor = StreamingFeatureExtractor(features=['crossing_points'], window_size=5)
    data_points = [1, 2, 3, 2, 1]
    for point in data_points:
        result = extractor.add_data(point)
    assert result['crossing_points']['crossing_count'] == 2, "Crossing count should be 2 for [1, 2, 3, 2, 1]"
    assert result['crossing_points']['crossing_points'] == [1, 3], "Crossing points should be [1, 3] for [1, 2, 3, 2, 1]"

def test_streaming_extractor_with_spikeness():
    """
    Test that StreamingFeatureExtractor calculates spikeness correctly.
    """
    extractor = StreamingFeatureExtractor(features=['spikeness'], window_size=5)
    data_points = [1, 10, 1, 10, 1]
    for point in data_points:
        result = extractor.add_data(point)
    assert result['spikeness'] > 0, "Spikeness should be greater than 0 for [1, 10, 1, 10, 1]"

def test_streaming_extractor_with_flat_spots():
    """
    Test that StreamingFeatureExtractor calculates flat spots correctly.
    """
    extractor = StreamingFeatureExtractor(features=['flat_spots'], window_size=5)
    data_points = [2, 2, 2, 3, 3]
    for point in data_points:
        result = extractor.add_data(point)
    assert result['flat_spots'] == 2, "Flat spots should be 2 for [2, 2, 2, 3, 3]"

def test_streaming_extractor_no_output_before_window():
    """
    Test that StreamingFeatureExtractor returns None before reaching window size.
    """
    extractor = StreamingFeatureExtractor(features=['mean'], window_size=4)
    data_points = [1, 2, 3]
    for point in data_points:
        result = extractor.add_data(point)
        assert result is None, "No output should be generated before reaching window size"

def test_streaming_extractor_with_multiple_features():
    """
    Test that StreamingFeatureExtractor calculates multiple features correctly.
    """
    extractor = StreamingFeatureExtractor(features=['mean', 'variance', 'peak'], window_size=3)
    data_points = [2, 4, 6]
    for point in data_points:
        result = extractor.add_data(point)
    assert result['mean'] == 4.0, "Mean should be 4.0 for [2, 4, 6]"
    assert result['variance'] == 4.0, "Variance should be 4.0 for [2, 4, 6]"
    assert result['peak'] == 6, "Peak should be 6 for [2, 4, 6]"

def test_streaming_extractor_with_rolling_window():
    """
    Test that StreamingFeatureExtractor calculates features correctly with rolling window.
    """
    extractor = StreamingFeatureExtractor(features=['mean'], window_size=3)
    data_points = [1, 2, 3, 4, 5]
    results = []
    for point in data_points:
        result = extractor.add_data(point)
        if result:
            results.append(result['mean'])
    assert results == [2.0, 3.0, 4.0], "Mean should be [2.0, 3.0, 4.0] with rolling window for data [1, 2, 3, 4, 5]"
