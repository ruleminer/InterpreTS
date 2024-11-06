import pandas as pd
import numpy as np
from src.core.features.seasonality_strength import calculate_seasonality_strength

def test_calculate_seasonality_strength_strong_seasonality():
    """
    Test that calculate_seasonality_strength identifies strong seasonality.
    This test uses a repeating seasonal pattern (e.g., a sine wave).
    """
    t = np.linspace(0, 10, 100)
    data = pd.Series(np.sin(t))
    result = calculate_seasonality_strength(data)
    assert result > 0.7, "The seasonality strength should be strong for periodic data"

def test_calculate_seasonality_strength_weak_seasonality():
    """
    Test that calculate_seasonality_strength identifies weak seasonality.
    This test uses a data set with some periodicity but lower correlation.
    """
    data = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 2.5, 1, 1.5, 3])
    result = calculate_seasonality_strength(data, period=3)
    assert result > 0.3, "The seasonality strength should be moderate for periodic data"

def test_calculate_seasonality_strength_no_seasonality():
    """
    Test that calculate_seasonality_strength identifies no seasonality.
    This test uses random data without any periodic pattern.
    """
    np.random.seed(0)
    data = pd.Series(np.random.randn(100))
    result = calculate_seasonality_strength(data)
    assert result < 0.2, "The seasonality strength should be low for random data"

def test_calculate_seasonality_strength_empty_series():
    """
    Test that calculate_seasonality_strength returns NaN for an empty series.
    """
    data = pd.Series([])
    result = calculate_seasonality_strength(data)
    assert pd.isna(result), "The seasonality strength of an empty series should be NaN"

def test_calculate_seasonality_strength_insufficient_data():
    """
    Test that calculate_seasonality_strength returns NaN for insufficient data.
    """
    data = pd.Series([5])
    result = calculate_seasonality_strength(data)
    assert pd.isna(result), "The seasonality strength should be NaN for insufficient data"

def test_calculate_seasonality_strength_with_short_periodicity():
    """
    Test that calculate_seasonality_strength detects seasonality with short periodicity.
    """
    data = pd.Series([1, 2, 3, 4, 5] * 3)
    result = calculate_seasonality_strength(data, period=5) 
    assert result > 0.5, "The seasonality strength should be significant for short periodicity data"

def test_calculate_seasonality_strength_with_long_periodicity():
    """
    Test that calculate_seasonality_strength detects seasonality with longer periodicity.
    """
    data = pd.Series([1, 0, 1, 0, 1, 0, 1, 0]) 
    result = calculate_seasonality_strength(data)
    assert result > 0.5, "The seasonality strength should be significant for longer periodicity data"

def test_calculate_seasonality_strength_numpy_array():
    """
    Test that calculate_seasonality_strength works with a numpy array.
    """
    data = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1])
    result = calculate_seasonality_strength(data, period=4)
    assert result > 0.5, "The seasonality strength should be significant for periodic data"
