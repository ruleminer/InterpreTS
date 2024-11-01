import pytest
import pandas as pd
from src.core.features.feature_length import calculate_length

def test_calculate_length():
    """
    Test the calculate_length function to ensure it correctly calculates
    the number of data points in a time series.
    """
    # Przykład danych
    data = pd.Series([1, 2, 3, 4, 5])

    # Sprawdzenie poprawności wyniku
    assert calculate_length(data) == 5, "The length of the time series should be 5"

def test_calculate_length_empty_series():
    """
    Test the calculate_length function with an empty time series.
    """
    data = pd.Series([])

    # Sprawdzenie poprawności wyniku dla pustego szeregu
    assert calculate_length(data) == 0, "The length of an empty time series should be 0"

def test_calculate_length_array():
    """
    Test the calculate_length function with a numpy array.
    """
    import numpy as np
    data = np.array([1, 2, 3, 4, 5])

    # Sprawdzenie poprawności wyniku dla numpy array
    assert calculate_length(data) == 5, "The length of the numpy array time series should be 5"
