# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.feature_absolute_energy import absolute_energy

# def test_absolute_energy():
#     """
#     Test that absolute_energy correctly calculates the absolute energy of a time series.
#     """
#     data = pd.Series([1, 2, 3, 4, 5])
#     assert absolute_energy(data) == 55.0, "The absolute energy should be 55.0"

# def test_absolute_energy_empty_series():
#     """
#     Test that absolute_energy returns NaN for an empty time series.
#     """
#     data = pd.Series([])
#     assert pd.isna(absolute_energy(data)), "The absolute energy of an empty series should be NaN"

# def test_absolute_energy_numpy_array():
#     """
#     Test that absolute_energy correctly calculates the absolute energy of a numpy array.
#     """
#     data = np.array([1, 2, 3, 4, 5])
#     assert absolute_energy(data) == 55.0, "The absolute energy should be 55.0 for numpy array"

# def test_absolute_energy_with_negative_values():
#     """
#     Test that absolute_energy correctly calculates the absolute energy for a series with negative values.
#     """
#     data = pd.Series([-1, -2, -3, -4, -5])
#     assert absolute_energy(data) == 55.0, "The absolute energy should be 55.0 even with negative values"

# def test_absolute_energy_with_start_and_end():
#     """
#     Test that absolute_energy correctly calculates the absolute energy within a specified start and end range.
#     """
#     data = pd.Series([1, 2, 3, 4, 5])
#     assert absolute_energy(data, start=1, end=4) == 29.0, "The absolute energy within index range 1 to 4 should be 29.0"

# def test_absolute_energy_with_out_of_range_indices():
#     """
#     Test that absolute_energy handles out-of-range start and end indices gracefully.
#     """
#     data = pd.Series([1, 2, 3, 4, 5])
#     assert pd.isna(absolute_energy(data, start=10)), "The absolute energy with out-of-range start index should be NaN"
#     assert pd.isna(absolute_energy(data, end=-10)), "The absolute energy with out-of-range end index should be NaN"

# def test_absolute_energy_with_start_greater_than_end():
#     """
#     Test that absolute_energy returns NaN if start index is greater than end index.
#     """
#     data = pd.Series([1, 2, 3, 4, 5])
#     assert pd.isna(absolute_energy(data, start=4, end=2)), "The absolute energy should be NaN if start index is greater than end index"

# def test_absolute_energy_with_all_zeros():
#     """
#     Test that absolute_energy correctly handles a series of all zeros.
#     """
#     data = pd.Series([0, 0, 0, 0, 0])
#     assert absolute_energy(data) == 0.0, "The absolute energy of a series of all zeros should be 0.0"

# def test_absolute_energy_with_single_element():
#     """
#     Test that absolute_energy correctly handles a series with a single element.
#     """
#     data = pd.Series([42])
#     assert absolute_energy(data) == 42**2, "The absolute energy of a single-element series should be the square of the element"

# def test_absolute_energy_with_floats():
#     """
#     Test that absolute_energy handles float values correctly.
#     """
#     data = pd.Series([1.5, 2.5, 3.5])
#     assert absolute_energy(data) == 1.5**2 + 2.5**2 + 3.5**2, "The absolute energy should correctly sum the squares of float values"
