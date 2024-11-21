import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_entropy import calculate_entropy

def test_calculate_entropy_with_valid_series():
    data = pd.Series([1, 2, 3, 4, 3, 6, 7, 6, 9, 10])
    entropy = calculate_entropy(data)
    assert 0 < entropy < 1, "Entropy should be between 0 and 1"

def test_calculate_entropy_with_constant_series():
    data = pd.Series([5, 5, 5, 5, 5])
    entropy = calculate_entropy(data)
    assert entropy == 0.0, "Entropy of a constant series should be 0"
    
def test_calculate_entropy_with_constant_changes():
    data = pd.Series([5, 6, 7, 8, 9])
    entropy = calculate_entropy(data)
    assert entropy == 1.0, "Entropy of a lineary changing series should be 1"
    
def test_calculate_entropy_with_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="The input data is empty."):
        calculate_entropy(data)

def test_calculate_entropy_with_short_series():
    data = pd.Series([1])
    with pytest.raises(ValueError, match="The input data is too short to calculate entropy."):
        calculate_entropy(data)