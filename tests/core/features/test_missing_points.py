import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_missing_points import calculate_missing_points 

# Test basic functionality with missing points
def test_missing_points_basic():
    data = pd.Series([1, 2, np.nan, 4, None])
    percentage_result = calculate_missing_points(data)
    count_result = calculate_missing_points(data, percentage=False)
    assert percentage_result == 0.4, "Percentage of missing points should be 40.0%"
    assert count_result == 2, "Missing points count should be 2"

# Test with no missing values
def test_missing_points_no_missing():
    data = pd.Series([1, 2, 3, 4, 5])
    percentage_result = calculate_missing_points(data)
    count_result = calculate_missing_points(data, percentage=False)
    assert percentage_result == 0.0, "Percentage of missing points should be 0.0%"
    assert count_result == 0, "Missing points count should be 0"

# Test where all values are missing
def test_missing_points_all_missing():
    data = pd.Series([np.nan, None, np.nan])
    percentage_result = calculate_missing_points(data)
    count_result = calculate_missing_points(data, percentage=False)
    assert percentage_result == 1.0, "Percentage of missing points should be 100.0%"
    assert count_result == 3, "Missing points count should be 3"

# Test with an empty series
def test_missing_points_empty_series():
    data = pd.Series([], dtype=float)
    percentage_result = calculate_missing_points(data)
    count_result = calculate_missing_points(data, percentage=False)
    assert np.isnan(percentage_result), "Percentage of missing points should be NaN for an empty series"
    assert np.isnan(count_result), "Missing points count should be NaN for an empty series"

# Test with a series containing mixed data types
def test_missing_points_mixed_types():
    data = pd.Series([1, "text", 3.5, np.nan, True])
    percentage_result = calculate_missing_points(data)
    count_result = calculate_missing_points(data, percentage=False)
    assert percentage_result == 0.2, "Percentage of missing points should be 20.0%"
    assert count_result == 1, "Missing points count should be 1"

# Test with a series containing a single valid value
def test_missing_points_only_one_value():
    data_valid = pd.Series([10])
    percentage_result_valid = calculate_missing_points(data_valid)
    count_result_valid = calculate_missing_points(data_valid, percentage=False)
    assert percentage_result_valid == 0.0, "Percentage of missing points should be 0.0% for a single valid value"
    assert count_result_valid == 0, "Missing points count should be 0 for a single valid value"

# Test with a series containing a single missing value
def test_missing_points_only_one_missing_value():
    data_missing = pd.Series([np.nan])
    percentage_result_missing = calculate_missing_points(data_missing)
    count_result_missing = calculate_missing_points(data_missing, percentage=False)
    assert percentage_result_missing == 1.0, "Percentage of missing points should be 100.0% for a single NaN value"
    assert count_result_missing == 1, "Missing points count should be 1 for a single NaN value"
