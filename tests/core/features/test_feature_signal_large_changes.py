import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_signal_large_changes import significant_change

def test_significant_change_with_valid_data():
    data = pd.Series([1, 2, 15, 3, 2.5, 5, 4.5])
    result = significant_change(data)
    assert result == 0.3333333333333333

def test_significant_change_with_no_significant_changes():
    data = pd.Series([1, 1.1, 1.2, 1.3, 1.4, 1.5])
    result = significant_change(data)
    assert result == 0.0

def test_significant_change_with_nan_values():
    data = pd.Series([1, 2, np.nan, 3, 4])
    with pytest.raises(ValueError):
        significant_change(data)

def test_significant_change_with_insufficient_data():
    data = pd.Series([1])
    result = significant_change(data)
    assert result == 0.0

def test_significant_change_with_non_time_series_data():
    data = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        significant_change(data)