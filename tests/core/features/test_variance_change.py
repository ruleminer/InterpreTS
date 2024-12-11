# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.variance_change import calculate_change_in_variance 

# def test_calculate_change_in_variance_constant_data():
#     """
#     Test that calculate_change_in_variance returns zeros for constant data.
#     """
#     data = pd.Series([5] * 10)  # Constant series
#     result = calculate_change_in_variance(data, window_size=3)
#     assert result.dropna().eq(0).all(), "Change in variance should be zero for constant data"

# def test_calculate_change_in_variance_increasing_variance():
#     """
#     Test that calculate_change_in_variance detects increasing variance.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 10, 20, 30, 40, 50])  # Increasing variability
#     result = calculate_change_in_variance(data, window_size=3)
#     assert result.dropna().iloc[-1] >= 0, "Change in variance should increase for growing variability"

# def test_calculate_change_in_variance_random_data():
#     """
#     Test that calculate_change_in_variance handles random data without errors.
#     """
#     np.random.seed(0)
#     data = pd.Series(np.random.randn(100))  # Random normal data
#     result = calculate_change_in_variance(data, window_size=5)
#     assert not result.isnull().all(), "Change in variance should not be all NaN for valid random data"

# def test_calculate_change_in_variance_empty_series():
#     """
#     Test that calculate_change_in_variance returns NaN for an empty series.
#     """
#     data = pd.Series([])
#     try:
#         result = calculate_change_in_variance(data, window_size=3)
#     except ValueError as e:
#         assert "too short" in str(e), "Function should raise a ValueError for empty series"

# def test_calculate_change_in_variance_insufficient_data():
#     """
#     Test that calculate_change_in_variance raises an error for insufficient data.
#     """
#     data = pd.Series([1, 2])
#     try:
#         result = calculate_change_in_variance(data, window_size=3)
#     except ValueError as e:
#         assert "too short" in str(e), "Function should raise a ValueError for insufficient data"

# def test_calculate_change_in_variance_window_size():
#     """
#     Test that calculate_change_in_variance respects the specified window size.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     window_size = 4
#     result = calculate_change_in_variance(data, window_size=window_size)
#     assert len(result.dropna()) == len(data) - window_size, "Result should have the correct length based on window size"

# def test_calculate_change_in_variance_numpy_array():
#     """
#     Test that calculate_change_in_variance works with a numpy array.
#     """
#     data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     result = calculate_change_in_variance(data, window_size=3)
#     assert isinstance(result, pd.Series), "Result should be a pandas Series for numpy input"
#     assert len(result) == len(data), "Result should have the same length as the input data"

# def test_calculate_change_in_variance_decreasing_variance():
#     """
#     Test that calculate_change_in_variance detects decreasing variance.
#     """
#     data = pd.Series([50, 40, 30, 20, 10, 5, 4, 3, 2, 1])  # Decreasing variability
#     result = calculate_change_in_variance(data, window_size=3)
#     assert result.dropna().iloc[-1] <= 0, "Change in variance should decrease for shrinking variability"
