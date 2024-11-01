import pytest
import pandas as pd
import numpy as np
from src.core.features.feature_length import calculate_length

def test_calculate_length_series():
    """
    Test the calculate_length function to ensure it correctly calculates
    the number of data points in a pandas Series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_length(data) == 5, "The length of the time series should be 5"

def test_calculate_length_empty_series():
    """
    Test the calculate_length function with an empty pandas Series.
    """
    data = pd.Series([])
    assert calculate_length(data) == 0, "The length of an empty series should be 0"

def test_calculate_length_array():
    """
    Test the calculate_length function with a numpy array.
    """
    data = np.array([1, 2, 3, 4, 5])
    assert calculate_length(data) == 5, "The length of the numpy array should be 5"

def test_calculate_length_empty_array():
    """
    Test the calculate_length function with an empty numpy array.
    """
    data = np.array([])
    assert calculate_length(data) == 0, "The length of an empty numpy array should be 0"

def test_calculate_length_dataframe():
    """
    Test the calculate_length function with a pandas DataFrame.
    """
    data = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    assert calculate_length(data) == 5, "The length of the DataFrame should be 5"
