import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_below_1st_decile import calculate_below_1st_decile

def test_calculate_below_1st_decile_valid_data():
    data = pd.Series([1, 2, 3, 4, 5])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_below_1st_decile(data, training_data)
    assert result == 0.2, "The fraction of values below the 1st decile should be 0.2"

def test_calculate_below_1st_decile_with_nan_values():
    data = pd.Series([1, np.nan, 3, 4, 5])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError):
        calculate_below_1st_decile(data, training_data)

def test_calculate_below_1st_decile_empty_data():
    data = pd.Series([])
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError):
        calculate_below_1st_decile(data, training_data)

def test_calculate_below_1st_decile_numpy_array():
    data = np.array([1, 2, 3, 4, 5])
    training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = calculate_below_1st_decile(data, training_data)
    assert result == 0.2, "The fraction of values below the 1st decile should be 0.2"

def test_calculate_below_1st_decile_invalid_type():
    data = "invalid data"
    training_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(TypeError):
        calculate_below_1st_decile(data, training_data)