# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.feature_flat_spots import calculate_flat_spots

# def test_calculate_flat_spots_with_flat_spots():
#     data = pd.Series([1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 1, 1])
#     result = calculate_flat_spots(data)
#     assert result == 4, "Expected 4 flat spots"

# def test_calculate_flat_spots_without_flat_spots():
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     result = calculate_flat_spots(data)
#     assert result == 1, "Expected 1 flat spot"

# def test_calculate_flat_spots_empty_series():
#     data = pd.Series([])
#     result = calculate_flat_spots(data)
#     assert result == 0, "Expected 0 flat spots for empty series"

# def test_calculate_flat_spots_constant_values():
#     data = pd.Series([5, 5, 5, 5, 5])
#     result = calculate_flat_spots(data)
#     assert result == 5, "Expected 10 flat spots for constant values"

# def test_calculate_flat_spots_different_window_size():
#     data = pd.Series([1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 1, 1])
#     result = calculate_flat_spots(data, window_size=3)
#     assert result == 3, "Expected 3 flat spots with window size 3"