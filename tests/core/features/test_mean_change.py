# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.mean_change import calculate_mean_change  

# def test_calculate_mean_change_constant_data():
#     """
#     Test that calculate_mean_change returns zeros for constant data.
#     """
#     data = pd.Series([5] * 10)  # Constant series
#     result = calculate_mean_change(data, window_size=3)
#     assert result.dropna().eq(0).all(), "Change in mean should be zero for constant data"

# def test_calculate_mean_change_increasing_data():
#     """
#     Test that calculate_mean_change detects increasing means.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Increasing trend
#     result = calculate_mean_change(data, window_size=3)
#     assert result.dropna().iloc[-1] > 0, "Change in mean should increase for growing data"

# def test_calculate_mean_change_decreasing_data():
#     """
#     Test that calculate_mean_change detects decreasing means.
#     """
#     data = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Decreasing trend
#     result = calculate_mean_change(data, window_size=3)
#     assert result.dropna().iloc[-1] < 0, "Change in mean should decrease for shrinking data"

# def test_calculate_mean_change_random_data():
#     """
#     Test that calculate_mean_change handles random data without errors.
#     """
#     np.random.seed(0)
#     data = pd.Series(np.random.randn(100))  # Random normal data
#     result = calculate_mean_change(data, window_size=5)
#     assert not result.isnull().all(), "Change in mean should not be all NaN for valid random data"

# def test_calculate_mean_change_empty_series():
#     """
#     Test that calculate_mean_change raises an error for an empty series.
#     """
#     data = pd.Series([])
#     with pytest.raises(ValueError, match="too short"):
#         calculate_mean_change(data, window_size=3)

# def test_calculate_mean_change_insufficient_data():
#     """
#     Test that calculate_mean_change raises an error for insufficient data.
#     """
#     data = pd.Series([1, 2])
#     with pytest.raises(ValueError, match="too short"):
#         calculate_mean_change(data, window_size=3)

# def test_calculate_mean_change_window_size():
#     """
#     Test that calculate_mean_change respects the specified window size.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     window_size = 4
#     result = calculate_mean_change(data, window_size=window_size)
#     assert len(result.dropna()) == len(data) - window_size, "Result should have the correct length based on window size"

# def test_calculate_mean_change_numpy_array():
#     """
#     Test that calculate_mean_change works with a numpy array.
#     """
#     data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     result = calculate_mean_change(data, window_size=3)
#     assert isinstance(result, pd.Series), "Result should be a pandas Series for numpy input"
#     assert len(result) == len(data), "Result should have the same length as the input data"