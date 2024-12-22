import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_flat_spots import calculate_flat_spots

# Test basic functionality of flat spots detection
def test_flat_spots_basic():
    data = pd.Series([1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 1, 1])
    result = calculate_flat_spots(data, window_size=5)
    assert result == 4, f"Expected 4, got {result}"

# Test for empty data input
def test_flat_spots_empty_data():
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_flat_spots(data)

# Test for data containing NaN values
def test_flat_spots_with_nan():
    data = pd.Series([1, 1, np.nan, 2, 3])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_flat_spots(data)

# Test for data without flat segments
def test_flat_spots_no_flat_segments():
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_flat_spots(data, window_size=3)
    assert result == 1, f"Expected 1, got {result}"

# Test for series where all values are equal
def test_flat_spots_all_values_equal():
    data = pd.Series([7, 7, 7, 7, 7])
    result = calculate_flat_spots(data, window_size=5)
    assert result == 5, f"Expected 5, got {result}"

# Test for varying window size impact
def test_flat_spots_varying_window_size():
    data = pd.Series([1, 1, 2, 2, 2, 3, 3, 3, 3])
    result = calculate_flat_spots(data, window_size=3)
    assert result == 3, f"Expected 3, got {result}"

# Test for input data as numpy array
def test_flat_spots_numpy_input():
    data = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    result = calculate_flat_spots(data, window_size=4)
    assert result == 3, f"Expected 3, got {result}"

# Test for data shorter than the window size
def test_flat_spots_short_data():
    data = pd.Series([1, 1, 1])
    result = calculate_flat_spots(data, window_size=5)
    assert result == 3, f"Expected 3, got {result}"
