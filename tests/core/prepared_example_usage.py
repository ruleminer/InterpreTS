from interpreTS.core.feature_extractor import FeatureExtractor, Features
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures

# Load the data (adjust the path if necessary)
file_path = 'lp1.data.txt'
timeseries, _ = load_robot_execution_failures(file_name='lp1.data.txt')
timeseries = timeseries[["id", "time", "F_x", "F_y"]]

# Print the first few rows of the data
print("Original Data:")
print(timeseries.head())

# Example 1: Extracting default features on the entire dataset
extractor = FeatureExtractor(id_column="id", sort_column="time")
features = extractor.extract_features(timeseries)
print("\nExtracted Features (Default Configuration):")
print(features.head())

# Example 2: Extracting specific features with a specified window size
extractor = FeatureExtractor(
    features=[Features.MEAN, Features.VARIANCE, Features.SPIKENESS],
    window_size=10,
    stride=5,
    id_column="id",
    sort_column="time"
)
features = extractor.extract_features(timeseries)
print("\nExtracted Features (Specific Features and Windowing):")
print(features.head())

# Example 3: Extracting features for a single column
extractor = FeatureExtractor(
    features=[Features.ENTROPY],
    feature_column="F_x",
    id_column="id",
    sort_column="F_x"
)
features = extractor.extract_features(timeseries)
print("\nExtracted Features (Single Column):")
print(features.head())

# Example 4: Handling edge cases (e.g., invalid parameters or missing data)
try:
    extractor = FeatureExtractor(features=[Features.MEAN], window_size=-5)
    features = extractor.extract_features(timeseries)
except ValueError as e:
    print("\nError with invalid window size:")
    print(e)

# Example 5: Grouping features by interpretability and generating descriptions
extractor = FeatureExtractor(features=[Features.LENGTH, Features.STABILITY, Features.MEAN])
features = extractor.extract_features(timeseries)

print("\nGrouped Features by Interpretability:")
print(extractor.group_features_by_interpretability())

# Limiting data to only ID, time, and one feature for demonstration
timeseries_limited = timeseries[["id", "F_x", "F_y"]]
extractor = FeatureExtractor(features=[Features.MEAN], sort_column="F_y", id_column="id")
features = extractor.extract_features(timeseries_limited)
print("\nExtracted Features (Limited Columns):")
print(features.head())
