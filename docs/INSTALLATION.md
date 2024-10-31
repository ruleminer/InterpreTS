# Installation Guide for InterpreTS
Follow the steps below to install InterpreTS and its dependencies.

## Requirements
Python 3.8 or higher
- [`pandas`](https://github.com/pandas-dev/pandas)
- [`numpy`](https://github.com/numpy/numpy)
  
## Steps
1. Clone the repository: Clone the InterpreTS repository to your local machine:
   
```python
git clone https://github.com/ruleminer/InterpreTS.git
cd time_series_feature_extractor
```

2. Install dependencies: Install the required packages listed in the `requirements.txt` file:

```python
pip install -r requirements.txt
```

3. Install InterpreTS: Run the following command to install InterpreTS:

```python
pip install .
```

## Verifying Installation
Once installed, you can verify the installation by running a simple feature extraction example:

```python
from interpreTS import FeatureExtractor
import pandas as pd

data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
extractor = FeatureExtractor(features=['length'])
features = extractor.extract_features(data)
print("Extracted Features:", features)
```

For any issues, please consult our [Issue Tracker](https://github.com/ruleminer/InterpreTS/issues) on GitHub.