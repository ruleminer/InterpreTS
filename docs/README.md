# InterpreTS - Overview 

 **interpreTS** is a Python library designed for extracting meaningful and interpretable features from time series data to support the creation of interpretable and explainable predictive models.

## Key Features
 - **Feature Extraction**: Extract features such as mean, variance, spikeness, entropy, trend strength, and more.
 - **Interpretable Models**: Generate explainable predictive models by leveraging extracted features.
 - **Streaming Data Support**: Process and extract features in real-time from streaming data sources.
 - **Scalability**: Supports parallel and distributed computation with `joblib` and `dask`.
 - **Custom Features**: Extend the library with user-defined features.
 - **Validation**: Ensures input data meets the required format and quality using built-in validators.

## Requirements
 - Python 3.10 or above
 - `pandas==2.2.3`
 - `numpy`
 - `statsmodels==0.14.4`
 - `langchain_community==0.3.14`
 - `langchain==0.3.14`
 - `openai==1.59.4`
 - `streamlit==1.41.1`
 - `scikit-learn`
 - `joblib==1.4.2`
 - `tqdm==4.67.1`
 - `dask==2024.12.1`
 - `scipy==1.15.0`
 - `pillow==11.1.0`

## Installation Guide
 Follow these steps to install InterpreTS and its dependencies:

### From PyPI
   
 ```bash
pip install interpreTS
 ```

### From Source
1. Clone the repository:
 ```bash
git clone https://github.com/yourusername/interpreTS.git
cd interpreTS
 ```

2. Install dependencies: Install the required packages listed in the `requirements.txt` file:

 ```bash
pip install -r requirements.txt
 ```

3. Install InterpreTS: Run the following command to install InterpreTS:

 ```bash
pip install .
 ```

## GUI
InterpreTS offers a user-friendly GUI to facilitate time series feature extraction and model explainability.

### Accessing the GUI
The GUI is hosted online and available at [InterpreTS GUI](https://ruleminer-interprets-interpretscoreguigui-streamlit-gui--zcpuxp.streamlit.app/). 

Alternatively, you can run it locally:
1. Navigate to the GUI directory:
 ```bash
cd interpreTS/core/gui
 ```

2. Launch the GUI using Streamlit:
 ```bash
streamlit run gui.py
 ```

### Features of the GUI
* Feature Extraction: Easily upload time series data and extract predefined or custom features.
* Visualization: View extracted features directly in the GUI.

## Verifying Installation - Example: Basic Feature Extraction
 Once installed, you can verify the installation by running a simple feature extraction example:

 ```python

import pandas as pd
from interpreTS import FeatureExtractor, Features

# Sample time series data
data = pd.DataFrame({
"time": pd.date_range("2023-01-01", periods=10, freq="D"),
"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# Initialize the FeatureExtractor
extractor = FeatureExtractor(features=[Features.MEAN, Features.VARIANCE], feature_column="value")

# Extract features
features_df = extractor.extract_features(data)
print(features_df)

 ```

## Additional Usage Example with Time Series Data

 ```python
import pandas as pd
import numpy as np
import time
from interpreTS import FeatureExtractor, Features

def report_progress(progress):
    print(f"Progress: {progress}%", flush=True)

# Generate synthetic time series data
data = pd.DataFrame({
'id': np.repeat(range(100), 100),  
'time': np.tile(range(100), 100),  
'value': np.random.randn(10000)  
})

# Initialize the FeatureExtractor
feature_extractor = FeatureExtractor(
features=[Features.ENTROPY],
feature_column="value",
id_column="id",
window_size=5,  # Rolling window size
stride=2        # Step size for moving the window
)

# Measure execution time
start_time = time.time()

# Extract features
features_df = feature_extractor.extract_features(data, progress_callback=report_progress, mode='sequential')

end_time = time.time()

# Display results and execution time
print(features_df.head())  # Display the first few rows of the resulting DataFrame
print(f"Execution time: {end_time - start_time:.2f} seconds")

 ```

## Documentation

Complete documentation is available on [GitHub Pages](https://ruleminer.github.io/InterpreTS/).


## Issues and Support

For any issues, please consult our [Issue Tracker](https://github.com/ruleminer/InterpreTS/issues) on GitHub.
