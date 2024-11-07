# InterpreTS

InterpreTS is a Python library designed for extracting meaningful and interpretable features from time series data to support the creation of interpretable and explainable predictive models.

## Overview
With the growing importance of interpretability in machine learning and AI, InterpreTS focuses on creating feature representations that facilitate the development of interpretable and explainable predictive models.

## Key Features
- **Statistical Features**: Extract basic statistics like mean, standard deviation, minimum, and maximum values.
- **Frequency Features**: Calculate features in the frequency domain, such as Fourier Transform coefficients.
- **Relational Features**: Generate features describing relationships between different time series, such as correlation.
- **Parallel Computing Support**: Efficiently compute features with parallel processing.
- **Data Format Flexibility**: Easily convert and process data in `pandas.DataFrame` or `numpy.array` formats.

## Requirements
- Python 3.8 or above
- `pandas`
- `numpy`
- `statsmodels`

## Installation Guide
Follow these steps to install InterpreTS and its dependencies:

1. **Clone the Repository**  
   Clone the InterpreTS repository to your local machine:
   
```bash
git clone https://github.com/ruleminer/InterpreTS.git
cd InterpreTS
```

2. Install dependencies: Install the required packages listed in the `requirements.txt` file:

 ```python
 pip install -r requirements.txt
 ```

3. Install InterpreTS: Run the following command to install InterpreTS:

 ```python
 pip install interpreTS
 ```

## Verifying Installation
Once installed, you can verify the installation by running a simple feature extraction example:

 ```python
 from interpreTS.core.feature_extractor import FeatureExtractor, Features
 import pandas as pd

 # Sample time series data
 data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
 extractor = FeatureExtractor(features=[Features.LENGTH, Features.MEAN, Features.VARIANCE])
 features = extractor.extract_features(data)
 print("Extracted Features:\n", features)
 ```

## Additional Usage Example with Time Series Data
You can also use InterpreTS with time-indexed data:

 ```python

 from interpreTS.core.time_series_data import TimeSeriesData
 from interpreTS.core.feature_extractor import FeatureExtractor, Features
 import pandas as pd

 # Time-indexed data
 data_with_date = pd.Series(
     [5, 3, 6, 2, 7, 4, 8, 3, 9, 1],
     index=pd.date_range("2023-01-01", periods=10, freq="D")
 )
 ts_data = TimeSeriesData(data_with_date)

 # Feature extraction
 extractor = FeatureExtractor(features=[Features.LENGTH, Features.MEAN, Features.VARIANCE])
 features = extractor.extract_features(ts_data.data)
 print("\nExtracted Features from Time Series Data:\n", features)
 ```

## Documentation

Complete documentation is available on [GitHub Pages](https://ruleminer.github.io/InterpreTS/).


## Issues and Support

For any issues, please consult our [Issue Tracker](https://github.com/ruleminer/InterpreTS/issues) on GitHub.
