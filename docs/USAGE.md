# Usage Guide for InterpreTS

This guide will walk you through using InterpreTS to extract features from time series data. Below are some examples demonstrating the primary functionalities of InterpreTS.

## Basic Feature Extraction

The following example demonstrates how to use FeatureExtractor to extract basic statistical and frequency-based features.

```python
from interpreTS import FeatureExtractor
import pandas as pd

# Sample time series data
data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

# Initialize the feature extractor with statistical and frequency features
extractor = FeatureExtractor(features=['length'])

# Extract features
features = extractor.extract_features(data)
print("Extracted Features:", features)

```