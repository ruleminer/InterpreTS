# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.feature_crossing_points import calculate_crossing_points

# def test_crossing_points_basic():
#     """
#     Test basic functionality with a typical time series.
#     """
#     data = pd.Series([1, 2, 3, 2, 1, 3, 1, 0])
#     result = calculate_crossing_points(data)
#     assert result['crossing_count'] == 4, "Crossing count should be 4"
#     assert result['crossing_points'] == [0, 3, 4, 5], "Crossing points should be [1, 4, 5, 6]"

# def test_crossing_points_one_crossing():
#     """
#     Test with a series that crosses its mean only once.
#     """
#     data = pd.Series([1, 2, 3, 4, 5])
#     result = calculate_crossing_points(data)
#     assert result['crossing_count'] == 1, "Crossing count should be 1"
#     assert result['crossing_points'] == [2], "Crossing points should be [2]"

# def test_crossing_points_negative_values():
#     """
#     Test with a series that includes negative values and crosses its mean multiple times.
#     """
#     data = pd.Series([-3, -1, 1, 3, -2, 2, -1, 1])
#     result = calculate_crossing_points(data)
#     assert result['crossing_count'] == 5, "Crossing count should be 5"
#     assert result['crossing_points'] == [1, 3, 4, 5, 6], "Crossing points should be [1, 3, 4, 5, 6]"

# def test_crossing_points_empty_series():
#     """
#     Test with an empty series to ensure it handles edge cases gracefully.
#     """
#     data = pd.Series([])
#     result = calculate_crossing_points(data)
#     assert result['crossing_count'] == 0, "Crossing count should be 0 for an empty series"
#     assert result['crossing_points'] == [], "Crossing points should be an empty list for an empty series"

# def test_crossing_points_all_above_mean():
#     """
#     Test a case where all values are well above the mean.
#     """
#     data = pd.Series([12, 12, 12, 12, 12])
#     result = calculate_crossing_points(data)
#     assert result['crossing_count'] == 0, "Crossing count should be 0 when all values are above mean"
#     assert result['crossing_points'] == [], "Crossing points should be an empty list when all values are above mean"

# def test_crossing_points_all_below_mean():
#     """
#     Test a case where all values are well below the mean.
#     """
#     data = pd.Series([-12, -12, -12, -12, -12])
#     result = calculate_crossing_points(data)
#     assert result['crossing_count'] == 0, "Crossing count should be 0 when all values are below mean"
#     assert result['crossing_points'] == [], "Crossing points should be an empty list when all values are below mean"
