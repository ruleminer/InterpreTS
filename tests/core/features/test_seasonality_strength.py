import pytest
import pandas as pd
import numpy as np
from src.core.features.seasonality_strength import calculate_seasonality_strength

def test_calculate_seasonality_strength_with_seasonality():
    """
    Test that calculate_seasonality_strength correctly identifies seasonality in a time series.
    """
    # Dane z wyraźną sezonowością (tygodniową)
    data = pd.Series([1, 2, 1, 2, 1, 2, 1] * 10, index=pd.date_range("2023-01-01", periods=70))
    strength = calculate_seasonality_strength(data, frequency=7)
    assert strength > 0.8, "The seasonality strength should be high for clearly seasonal data"

def test_calculate_seasonality_strength_no_seasonality():
    """
    Test that calculate_seasonality_strength returns low seasonality strength for non-seasonal data.
    """
    data = pd.Series(np.arange(100), index=pd.date_range("2023-01-01", periods=100))
    strength = calculate_seasonality_strength(data, frequency=7)
    assert strength < 0.1, "The seasonality strength should be low for non-seasonal data"

def test_calculate_seasonality_strength_empty_series():
    """
    Test that calculate_seasonality_strength returns 0 for an empty time series.
    """
    data = pd.Series([], dtype=float)
    strength = calculate_seasonality_strength(data, frequency=7)
    assert strength == 0, "The seasonality strength should be 0 for an empty series"

def test_calculate_seasonality_strength_single_value():
    """
    Test that calculate_seasonality_strength returns 0 for a time series with a single value.
    """
    data = pd.Series([5], index=pd.date_range("2023-01-01", periods=1))
    strength = calculate_seasonality_strength(data, frequency=7)
    assert strength == 0, "The seasonality strength should be 0 for a single-value series"

def test_calculate_seasonality_strength_specified_frequency():
    """
    Test that calculate_seasonality_strength correctly uses the specified frequency.
    """
    data = pd.Series([1, 2, 3, 2] * 25, index=pd.date_range("2023-01-01", periods=100))
    strength_weekly = calculate_seasonality_strength(data, frequency=7)
    strength_monthly = calculate_seasonality_strength(data, frequency=30)
    assert strength_monthly > strength_weekly, "Monthly seasonality strength should be higher than weekly"
