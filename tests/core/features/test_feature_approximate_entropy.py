import pytest
import numpy as np
from interpreTS.core.features.feature_approximate_entropy import calculate_approximate_entropy

def test_approximate_entropy_basic_case():
    data = [1, 2, 3, 4, 5]
    result = calculate_approximate_entropy(data)
    assert not np.isnan(result), "ApEn should not be NaN for a simple series"

def test_approximate_entropy_constant_data():
    data = [5, 5, 5, 5, 5]
    result = calculate_approximate_entropy(data)
    assert result == 0.0, "ApEn should be 0 for constant data"

def test_approximate_entropy_high_variance():
    data = [1, 5, 9, 4, 3, 8, 6]
    result = calculate_approximate_entropy(data)
    assert result >= -5, "ApEn should not be significantly negative for random-like data"

def test_approximate_entropy_low_variance():
    data = [1, 1.1, 1.2, 1.1, 1.0]
    result = calculate_approximate_entropy(data)
    assert result < 1, "ApEn should be low for highly regular data"

def test_approximate_entropy_with_custom_params():
    data = [1, 3, 5, 7, 9]
    result = calculate_approximate_entropy(data, m=3, r=0.2)
    assert result >= -5, "ApEn should not be significantly negative for data with some variance"
