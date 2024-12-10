import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_above_9th_decile import calculate_above_9th_decile

def test_calculate_above_9th_decile_with_valid_data():
    data = pd.Series([1, 8, 9, 10, 11])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 0.4

def test_calculate_above_9th_decile_with_no_values_above_decile():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 0.0

def test_calculate_above_9th_decile_with_all_values_above_decile():
    data = pd.Series([11, 12, 13, 14, 15])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_above_9th_decile(data, training_data)
    assert result == 1.0

def test_calculate_above_9th_decile_with_nan_values():
    data = pd.Series([1, 8, 9, np.nan, 11])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError):
        calculate_above_9th_decile(data, training_data)

def test_calculate_above_9th_decile_with_invalid_data_type():
    data = [1, 8, 9, 10, 11]
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(TypeError):
        calculate_above_9th_decile(data, training_data)

def test_calculate_above_9th_decile_with_empty_data():
    data = pd.Series([])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError):
        calculate_above_9th_decile(data, training_data)