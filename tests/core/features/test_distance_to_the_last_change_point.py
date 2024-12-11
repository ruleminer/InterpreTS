# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.distance_to_the_last_change_point import calculate_distance_to_last_trend_change  

# def test_calculate_distance_to_last_trend_change_constant_data():
#     """
#     Test that calculate_distance_to_last_trend_change returns None for constant data (no trend change).
#     """
#     data = pd.Series([5] * 10)  # Constant series
#     result = calculate_distance_to_last_trend_change(data, window_size=3)
#     assert result is None, "There should be no trend change in constant data"


# def test_calculate_distance_to_last_trend_change_increasing_data():
#     """
#     Test that calculate_distance_to_last_trend_change detects the correct distance for increasing data.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Increasing trend
#     result = calculate_distance_to_last_trend_change(data, window_size=3)
#     assert result is None, "No trend change should occur for consistently increasing data"


# def test_calculate_distance_to_last_trend_change_decreasing_data():
#     """
#     Test that calculate_distance_to_last_trend_change detects the correct distance for decreasing data.
#     """
#     data = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Decreasing trend
#     result = calculate_distance_to_last_trend_change(data, window_size=3)
#     assert result is None, "No trend change should occur for consistently decreasing data"


# def test_calculate_distance_to_last_trend_change_random_data():
#     """
#     Test that calculate_distance_to_last_trend_change handles random data and detects the last trend change.
#     """
#     np.random.seed(0)
#     data = pd.Series(np.random.randn(100))  # Random normal data
#     result = calculate_distance_to_last_trend_change(data, window_size=5)
#     assert isinstance(result, int), "The result should be an integer representing the distance"
#     assert result >= 0, "Distance to last trend change should be a non-negative integer"


# def test_calculate_distance_to_last_trend_change_no_trend_change():
#     """
#     Test that calculate_distance_to_last_trend_change returns None when there is no trend change.
#     """
#     data = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Constant data
#     result = calculate_distance_to_last_trend_change(data, window_size=3)
#     assert result is None, "There should be no trend change in constant data"


# def test_calculate_distance_to_last_trend_change_single_trend_change():
#     """
#     Test that calculate_distance_to_last_trend_change detects the correct distance to the last trend change.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])  # Trend change after index 5
#     result = calculate_distance_to_last_trend_change(data, window_size=3)
#     assert result == 8, "Distance to the last trend change should be 8"


# def test_calculate_distance_to_last_trend_change_multiple_trend_changes():
#     """
#     Test that calculate_distance_to_last_trend_change detects the correct distance to the last trend change 
#     after multiple changes.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7])  # Multiple trend changes
#     result = calculate_distance_to_last_trend_change(data, window_size=3)
#     assert result == 10, "Distance to the last trend change should be 10 (last change is at index 10)"


# def test_calculate_distance_to_last_trend_change_empty_series():
#     """
#     Test that calculate_distance_to_last_trend_change raises an error for an empty series.
#     """
#     data = pd.Series([])
#     with pytest.raises(ValueError, match="too short"):
#         calculate_distance_to_last_trend_change(data, window_size=3)


# def test_calculate_distance_to_last_trend_change_insufficient_data():
#     """
#     Test that calculate_distance_to_last_trend_change raises an error for insufficient data.
#     """
#     data = pd.Series([1, 2])
#     with pytest.raises(ValueError, match="too short"):
#         calculate_distance_to_last_trend_change(data, window_size=3)


# def test_calculate_distance_to_last_trend_change_window_size():
#     """
#     Test that calculate_distance_to_last_trend_change respects the specified window size.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Increasing trend
#     window_size = 4
#     result = calculate_distance_to_last_trend_change(data, window_size=window_size)
#     if data.is_monotonic_increasing or data.is_monotonic_decreasing:
#         assert result is None, "For monotonic data, result should be None"
#     else:
#         assert isinstance(result, int), "Result should be an integer representing the distance to the last trend change"



# def test_calculate_distance_to_last_trend_change_numpy_array():
#     """
#     Test that calculate_distance_to_last_trend_change works with a numpy array.
#     """
#     data = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
#     result = calculate_distance_to_last_trend_change(data, window_size=3)
#     assert isinstance(result, int), "Result should be an integer for numpy array input"
#     assert result >= 0, "Distance should be a non-negative integer"