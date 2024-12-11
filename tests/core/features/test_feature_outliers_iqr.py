# import numpy as np
# from interpreTS.core.features.feature_outliers_iqr import calculate_outliers_iqr, FeatureExtractor, Features


# def test_calculate_outliers_iqr():
#     training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#     window_data = np.array([1, 20, 3, -10, 5])

#     result = calculate_outliers_iqr(window_data, training_data)
    
#     # Expected result: 40% out (2 in 5 points)
#     assert np.isclose(result, 0.4), f"Expected 0.4, got {result}"

# def test_feature_extractor_with_outliers_iqr():
#     training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#     window_data = np.array([1, 20, 3, -10, 5])

#     extractor = FeatureExtractor(features=[Features.OUTLIERS_IQR], training_data=training_data)
#     features = extractor.extract_features(window_data)

#     assert len(features) == 1, "Expected 1 feature"
#     assert np.isclose(features[Features.OUTLIERS_IQR], 0.4), f"Expected 0.4, got {features[Features.OUTLIERS_IQR]}"
