import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_spikeness import calculate_spikeness

def test_spikeness_positive_skew():
    """
    Test calculate_spikeness on a positively skewed time series.
    """
    data = pd.Series([1, 2, 3, 4, 10])
    spikeness = calculate_spikeness(data)
    assert spikeness > 0, "Spikeness should be positive for positively skewed data"

def test_spikeness_negative_skew():
    """
    Test calculate_spikeness on a negatively skewed time series.
    """
    data = pd.Series([50, 40, 30, 20, 10, 5, 2, 1, 0, -20, -30, -40]) 
    spikeness = calculate_spikeness(data)
    assert spikeness < 0, "Spikeness should be negative for negatively skewed data"
    
def test_spikeness_no_skew():
    """
    Test calculate_spikeness on a time series with no skew (symmetric data).
    """
    data = pd.Series([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])  # Większy symetryczny rozkład
    spikeness = calculate_spikeness(data)
    assert np.isclose(spikeness, 0, atol=1e-1), "Spikeness should be close to 0 for symmetric data"

def test_spikeness_empty_series():
    """
    Test calculate_spikeness on an empty time series.
    """
    data = pd.Series([], dtype="float64")
    spikeness = calculate_spikeness(data)
    assert np.isnan(spikeness), "Spikeness should be NaN for an empty series"

def test_spikeness_with_nan_values():
    """
    Test calculate_spikeness on a time series containing NaN values.
    """
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values"):
        calculate_spikeness(data)

def test_spikeness_numpy_array():
    """
    Test calculate_spikeness on a numpy array (alternative data format).
    """
    data = np.array([1, 2, 3, 4, 10])
    spikeness = calculate_spikeness(pd.Series(data))
    assert spikeness > 0, "Spikeness should be positive for positively skewed data"
