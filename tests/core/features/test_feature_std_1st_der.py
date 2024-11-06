import pytest
import pandas as pd
import numpy as np
from src.core.features.feature_std_1st_der import calculate_std_1st_der

def test_calculate_std_1st_derivative_constant_series():
    """
    Test that the standard deviation of the first derivative is zero for a constant series.
    """
    data = pd.Series([5, 5, 5, 5, 5])
    assert calculate_std_1st_der(data) == 0.0, "The std of first derivative for a constant series should be 0"

def test_calculate_std_1st_derivative_linear_series():
    """
    Test that the standard deviation of the first derivative is zero for a linear series.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_std_1st_der(data) == 0.0, "The std of first derivative for a linear series should be 0"

def test_calculate_std_1st_derivative_varying_series():
    """
    Test the standard deviation of the first derivative for a varying series.
    """
    data = pd.Series([1, 2, 1, 2, 1])
    result = 0.63245553
    assert calculate_std_1st_der(data) == pytest.approx(result, rel=1e-5), f"The std of first derivative should be approximately {result:.5f})"

def test_calculate_std_1st_derivative_numpy_array():
    """
    Test that the function works with a numpy array as input.
    """
    data = np.array([1, -1, 2, -3, 5])
    result = 3.5128336
    assert calculate_std_1st_der(data) == pytest.approx(result, rel=1e-5), f"The std of first derivative should be approximately {result:.5f})"
