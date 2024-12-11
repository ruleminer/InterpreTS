# import pytest
# import pandas as pd
# import numpy as np
# from interpreTS.core.features.feature_stability import calculate_stability

# def test_stability_high_stability():
#     """
#     Test calculate_stability on a highly stable time series.
#     """
#     data = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#     stability = calculate_stability(data, 2)
#     assert stability == 1.0, "Stability should be 1 for a perfectly stable series"

# def test_stability_low_stability():
#     """
#     Test calculate_stability on a low stability time series.
#     """
#     data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#     stability = calculate_stability(data)
#     assert stability < 1.0, "Stability should be less than 1 for a non-stable series"

# def test_two_stabilities():
#     """
#     Test calculate_stability on two different time series.
#     """
#     data1 = pd.Series([1, 2, 1, 1, 1, 1, 1, 1])
#     data2 = pd.Series([1, 2, 1, 2, 1, 2, 1, 1])
#     assert calculate_stability(data1) > calculate_stability(data2), "Stability should be higher for the first series"

# def test_stability_empty_series():
#     """
#     Test calculate_stability on an empty time series.
#     """
#     data = pd.Series([], dtype="float64")
#     stability = calculate_stability(data)
#     assert np.isnan(stability), "Stability should be NaN for an empty series"

# def test_stability_with_nan_values():
#     """
#     Test calculate_stability on a time series containing NaN values.
#     """
#     data = pd.Series([1, 2, np.nan, 4, 5])
#     with pytest.raises(ValueError, match="Data contains NaN values"):
#         calculate_stability(data)

# def test_stability_numpy_array():
#     """
#     Test calculate_stability on a numpy array (alternative data format).
#     """
#     data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#     stability = calculate_stability(pd.Series(data))
#     assert stability == 1.0, "Stability should be 1 for a perfectly stable series"