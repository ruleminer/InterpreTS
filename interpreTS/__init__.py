"""
interpreTS: A library for time series analysis and feature extraction.

Available imports:
- FeatureExtractor: Extracts features from time series data.
- StreamingFeatureExtractor: Processes streaming time series data.
- Features: Contains constants for supported feature calculations.
- TimeSeriesData: Utility class for handling and manipulating time series data.
- convert_to_time_series: Converts raw data into a time series format.
- validate_time_series_data: Validates the integrity and format of time series data.

Dependencies:
- pandas>=1.1.0
- numpy>=1.18.0
- statsmodels>=0.12.0
- pytest
"""

from .core.streaming_feature_extractor import StreamingFeatureExtractor
from .core.feature_extractor import FeatureExtractor
from .core.feature_extractor import Features
from .core.time_series_data import TimeSeriesData
from .utils.data_conversion import convert_to_time_series
from .utils.data_validation import validate_time_series_data

__version__ = "0.4.0"

__all__ = [
    "FeatureExtractor",
    "StreamingFeatureExtractor",
    "Features",
    "TimeSeriesData",
    "convert_to_time_series",
    "validate_time_series_data",
]

# Check required dependencies
required_libraries = {
    "pandas": "1.1.0",
    "numpy": "1.18.0",
    "statsmodels": "0.12.0",
    "pytest": None,  
}

for library, min_version in required_libraries.items():
    try:
        module = __import__(library)
        if min_version:
            from packaging import version
            if version.parse(module.__version__) < version.parse(min_version):
                print(f"Warning: {library} version must be >= {min_version}. Current version: {module.__version__}")
    except ImportError:
        print(f"Warning: {library} is not installed. Please install it to use interpreTS.")
