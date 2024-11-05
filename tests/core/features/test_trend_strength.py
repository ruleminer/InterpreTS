import pytest
import pandas as pd
import numpy as np
from src.core.features.trend_strength import calculate_trend_strength

def test_calculate_trend_strength_increasing():
    """
    Test that calculate_trend_strength identifies a strong positive trend.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_trend_strength(data) == pytest.approx(1.0), "The trend strength should be close to 1.0 for a perfect increasing trend"

def test_calculate_trend_strength_decreasing():
    """
    Test that calculate_trend_strength identifies a strong negative trend.
    """
    data = pd.Series([5, 4, 3, 2, 1])
    assert calculate_trend_strength(data) == pytest.approx(1.0), "The trend strength should be close to 1.0 for a perfect decreasing trend"

def test_calculate_trend_strength_no_trend():
    """
    Test that calculate_trend_strength returns a low value for a series with no trend.
    """
    data = pd.Series([1, 1, 1, 1, 1])
    assert calculate_trend_strength(data) == pytest.approx(0.0), "The trend strength should be 0.0 for no trend"

def test_calculate_trend_strength_random_data():
    """
    Test that calculate_trend_strength returns a low value for random data.
    """
    np.random.seed(0)
    data = pd.Series(np.random.randn(100))
    assert calculate_trend_strength(data) < 0.2, "The trend strength should be low for random data"

def test_calculate_trend_strength_empty_series():
    """
    Test that calculate_trend_strength returns NaN for an empty series.
    """
    data = pd.Series([])
    assert pd.isna(calculate_trend_strength(data)), "The trend strength of an empty series should be NaN"

def test_calculate_trend_strength_insufficient_data():
    """
    Test that calculate_trend_strength returns NaN for a single data point.
    """
    data = pd.Series([5])
    assert pd.isna(calculate_trend_strength(data)), "The trend strength should be NaN for insufficient data"

def test_calculate_trend_strength_numpy_array():
    """
    Test that calculate_trend_strength works with a numpy array.
    """
    data = np.array([1, 2, 3, 4, 5])
    assert calculate_trend_strength(data) == pytest.approx(1.0), "The trend strength should be close to 1.0 for a perfect trend"