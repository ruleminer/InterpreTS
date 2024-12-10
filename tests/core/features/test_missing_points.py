import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_missing_points import missing_points

def test_missing_points_basic():
    """
    Test basic functionality with a typical time series.
    """
    data = pd.Series([1, 2, np.nan, 4, None, 5])
    percentage_result = missing_points(data)
    count_result = missing_points(data, percentage=False)
    assert percentage_result == 0.3333333333333333, "Percentage of missing points should be ~33.33%"
    assert count_result == 2, "Missing points count should be 2"

def test_missing_points_no_missing():
    """
    Test with a series that has no missing values.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    percentage_result = missing_points(data)
    count_result = missing_points(data, percentage=False)
    assert percentage_result == 0.0, "Percentage of missing points should be 0%"
    assert count_result == 0, "Missing points count should be 0"

def test_missing_points_all_missing():
    """
    Test with a series where all values are missing.
    """
    data = pd.Series([np.nan, None, np.nan])
    percentage_result = missing_points(data)
    count_result = missing_points(data, percentage=False)
    assert percentage_result == 1.0, "Percentage of missing points should be 100%"
    assert count_result == 3, "Missing points count should be 3"

def test_missing_points_empty_series():
    """
    Test with an empty series to ensure it handles edge cases gracefully.
    """
    data = pd.Series([])
    percentage_result = missing_points(data)
    count_result = missing_points(data, percentage=False)
    assert np.isnan(percentage_result), "Percentage of missing points should be NaN for an empty series"
    assert np.isnan(count_result), "Missing points count should be NaN for an empty series"

def test_missing_points_mixed_types():
    """
    Test with a series containing mixed types including NaN values.
    """
    data = pd.Series([1, "text", None, 3.5, np.nan, True])
    percentage_result = missing_points(data)
    count_result = missing_points(data, percentage=False)
    assert percentage_result == 0.3333333333333333, "Percentage of missing points should be ~33.33%"
    assert count_result == 2, "Missing points count should be 2"

def test_missing_points_only_one_value():
    """
    Test with a series containing only one value, either valid or NaN.
    """
    data_valid = pd.Series([10])

    percentage_result_valid = missing_points(data_valid)
    count_result_valid = missing_points(data_valid, percentage=False)
    assert percentage_result_valid == 0.0, "Percentage of missing points should be 0% for a single valid value"
    assert count_result_valid == 0, "Missing points count should be 0 for a single valid value"

def test_missing_points_only_one_missing_value():
    data_missing = pd.Series([np.nan])
        
    percentage_result_missing = missing_points(data_missing)
    count_result_missing = missing_points(data_missing, percentage=False)
    assert percentage_result_missing == 1.0, "Percentage of missing points should be 100% for a single NaN value"
    assert count_result_missing == 1, "Missing points count should be 1 for a single NaN value"
