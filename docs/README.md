# InterpreTS

Feature extraction from time series to support the creation of interpretable and explainable predictive models.

## Overview
InterpreTS is a Python library designed for extracting meaningful and interpretable features from time series data. With the growing importance of interpretability in machine learning and AI, InterpreTS focuses on creating feature representations that facilitate the development of interpretable and explainable predictive models.

## Key Features
- **Statistical Features**: Extract basic statistics like mean, standard deviation, minimum, and maximum values.
- **Frequency Features**: Calculate features in the frequency domain, such as Fourier Transform coefficients.
- **Relational Features**: Generate features describing relationships between different time series, such as correlation.
- **Parallel Computing Support**: Efficiently compute features with parallel processing.
- **Data Format Flexibility**: Easily convert and process data in `pandas.DataFrame` or `numpy.array` formats.

## Requirements
- Python 3.8 or above
- pandas
- numpy

## Installation
To install InterpreTS, follow the steps in [Installation Guide](docs/INSTALLATION.md).

## Quickstart
To begin using InterpreTS, see the [Usage Guide](docs/USAGE.md) or follow the example below:

```python
from interpreTS import FeatureExtractor, TimeSeriesData
import pandas as pd

# Sample data
data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

# Initialize the feature extractor with desired features
extractor = FeatureExtractor(features=['length'])

# Extract features
features = extractor.extract_features(data)
print("Extracted Features:", features)
