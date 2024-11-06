
import pytest
import pandas as pd
import numpy as np
from src.core.features.feature_spikness import calculate_spikeness

def test_calculate_spikness():
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_spikness(data) == 0.0, "The spikness should be 0.0 for a symmetric series"

def test_calculate_spikness_empty_series():
    data = pd.Series([])
    assert pd.isna(calculate_spikness(data)), "The spikness of an empty series should be NaN"

def test_calculate_spikness_with_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    assert calculate_spikness(pd.Series(data)) == 0.0, "The spikness should be 0.0 for a symmetric numpy array"

def test_calculate_spikness_negative_skew():
    data = pd.Series([5, 4, 3, 2, 1])
    assert calculate_spikness(data) < 0, "The spikness should be negative for left-skewed data"
