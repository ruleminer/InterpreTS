import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_absolute_energy import absolute_energy  # Zmień ścieżkę do modułu, jeśli to konieczne

# Test absolute energy for the entire series
def test_absolute_energy_full_series():
    data = pd.Series([1, 2, 3, 4])
    result = absolute_energy(data)
    assert result == 30.0, "Absolute energy calculation failed for the full series"

# Test absolute energy for a specified range (start and end)
def test_absolute_energy_with_start_and_end():
    data = pd.Series([1, 2, 3, 4])
    result = absolute_energy(data, start=1, end=3)
    assert result == 13.0, "Absolute energy calculation failed for specified range"

# Test absolute energy with only the start specified
def test_absolute_energy_with_start_only():
    data = pd.Series([1, 2, 3, 4])
    result = absolute_energy(data, start=2)
    assert result == 25.0, "Absolute energy calculation failed when only start is specified"

# Test absolute energy with only the end specified
def test_absolute_energy_with_end_only():
    data = pd.Series([1, 2, 3, 4])
    result = absolute_energy(data, end=2)
    assert result == 5.0, "Absolute energy calculation failed when only end is specified"

# Test absolute energy for an empty series
def test_absolute_energy_empty_series():
    data = pd.Series([])
    with pytest.raises(ValueError, match="Input data is empty."):
        absolute_energy(data)

# Test absolute energy for a single value
def test_absolute_energy_single_value():
    data = pd.Series([5])
    result = absolute_energy(data)
    assert result == 25.0, "Absolute energy calculation failed for a single value"

# Test absolute energy with a range out of bounds
def test_absolute_energy_start_end_out_of_bounds():
    data = pd.Series([1, 2, 3, 4])
    result = absolute_energy(data, start=10, end=15)
    assert np.isnan(result), "Absolute energy calculation should return NaN for out-of-bounds range"

# Test absolute energy when start is greater than end
def test_absolute_energy_start_greater_than_end():
    data = pd.Series([1, 2, 3, 4])
    result = absolute_energy(data, start=3, end=2)
    assert np.isnan(result), "Absolute energy calculation should return NaN when start > end"

# Test absolute energy with NaN values in the series
def test_absolute_energy_nan_values():
    data = pd.Series([1, np.nan, 3, 4])
    result = absolute_energy(data)
    assert result == 26.0, "Absolute energy calculation can handle NaN values"

# Test absolute energy for numpy array input
def test_absolute_energy_numpy_array():
    data = np.array([1, 2, 3, 4])
    result = absolute_energy(data)
    assert result == 30.0, "Absolute energy calculation failed for numpy array"

# Test absolute energy for a large dataset
def test_absolute_energy_large_dataset():
    data = pd.Series(range(1, 10001))
    result = absolute_energy(data)
    expected = sum(i**2 for i in range(1, 10001))
    assert result == expected, "Absolute energy calculation failed for a large dataset"
