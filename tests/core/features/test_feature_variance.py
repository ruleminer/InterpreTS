# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.feature_variance import calculate_variance

# def test_calculate_variance():
#     """
#     Test that calculate_variance correctly calculates the variance of a time series.
#     """
#     data = pd.Series([1, 2, 3, 4, 5])
#     assert calculate_variance(data) == 2.5, "The variance value should be 2.5"

# def test_calculate_variance_empty_series():
#     """
#     Test that calculate_variance returns NaN for an empty time series.
#     """
#     data = pd.Series([])
#     assert pd.isna(calculate_variance(data)), "The variance of an empty series should be NaN"

# def test_calculate_variance_numpy_array():
#     """
#     Test that calculate_variance correctly calculates the variance of a numpy array.
#     """
#     data = np.array([1, 2, 3, 4, 5])
#     assert calculate_variance(data) == 2.5, "The variance should be 2.5 for numpy array"

# def test_calculate_variance_negative_values():
#     """
#     Test that calculate_variance correctly calculates the variance for a series with negative values.
#     """
#     data = pd.Series([-1, -2, -3, -4, -5])
#     assert calculate_variance(data) == 2.5, "The mean value should be 2.0"

# def test_calculate_variance_with_ddof_0():
#     """
#     Test that calculate_variance correctly calculates the population variance (ddof=0).
#     """
#     data = pd.Series([1, 2, 3, 4, 5])
#     assert calculate_variance(data, ddof=0) == 2.0, "The population variance should be 2.0"

# def test_calculate_variance_with_single_value():
#     """
#     Test that calculate_variance returns 0 for a series with a single value.
#     """
#     data = pd.Series([42])
#     assert calculate_variance(data) == 0.0, "The variance of a single value series should be 0.0"

# def test_calculate_variance_with_identical_values():
#     """
#     Test that calculate_variance returns 0 for a series with identical values.
#     """
#     data = pd.Series([5, 5, 5, 5, 5])
#     assert calculate_variance(data) == 0.0, "The variance of identical values should be 0.0"