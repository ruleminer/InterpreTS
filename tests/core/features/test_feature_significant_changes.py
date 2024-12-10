import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_significant_changes import calculate_significant_changes

def test_calculate_significant_changes_with_valid_data():
    data = pd.Series([1, 2, 15, 3, 2.5, 5, 4.5])
    result = calculate_significant_changes(data)
    assert result == 0.3333333333333333

def test_calculate_significant_changes_with_no_significant_changes():
    data = pd.Series([1, 1.1, 1.2, 1.3, 1.4, 1.5])
    result = calculate_significant_changes(data)
    assert result == 0.0

def test_calculate_significant_changes_with_nan_values():
    data = pd.Series([1, 2, np.nan, 3, 4])
    with pytest.raises(ValueError):
        calculate_significant_changes(data)

def test_calculate_significant_changes_with_insufficient_data():
    data = pd.Series([1])
    result = calculate_significant_changes(data)
    assert result == 0.0

def test_calculate_significant_changes_with_non_time_series_data():
    data = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        calculate_significant_changes(data)