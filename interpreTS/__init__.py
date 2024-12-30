"""
interpreTS: Python library designed for extracting meaningful and interpretable features from time series data to support the creation of interpretable and explainable predictive models.

Available imports:
    FeatureExtractor (from interpreTS.core.feature_extractor):
        A class responsible for extracting specified features from time series data.
        
    Features (from interpreTS.utils.feature_loader):
        An enumeration or collection that defines available feature types for extraction.
         
    FeatureLoader (from interpreTS.utils.feature_loader):
        A utility class for loading and managing feature definitions.
    
    validate_time_series_data (from interpreTS.utils.data_validation):
        A function to ensure that input time series data meets the required format and standards for processing.
    
    generate_feature_descriptions (from interpreTS.utils.data_manager):
        A function that generates human-readable descriptions for extracted features, aiding interpretability.
        
Dependencies:
    - pandas>=1.1.0
    - numpy>=1.18.0
    - statsmodels>=0.12.0
    - langchain_community
    - langchain 
    - openai
    - scikit-learn
    - joblib
    - tqdm
    - dask
    - nbsphinx
    - myst-parser
    - scipy
    
Authors:
    - Sławomir Put,
    - Martyna Żur,
    - Weronika Wołowczyk
    - Jarosław Strzelczyk,
    - Piotr Krupiński
    - Martyna Kramarz
    - Łukasz Wróbel

"""
import sys
import logging
from packaging import version


# Set up logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Check Python version
if sys.version_info < (3, 8):
    raise RuntimeError("interpreTS requires Python 3.8 or higher.")

# Check required dependencies
required_libraries = {
    "pandas": None,
    "numpy": None,
    "statsmodels": None,
    "streamlit": None,
    "langchain_community" : None,
    "langchain" : None,
    "openai" : None,
    "scikit-learn" : None,
    "joblib" : None,
    "tqdm" : None,
    "dask" : None,
    "nbsphinx" : None,
    "myst-parser" : None,
    "scipy" : None
}

for library, min_version in required_libraries.items():
    try:
        module = __import__(library)
        if min_version and version.parse(module.__version__) < version.parse(min_version):
            logger.warning(f"{library} version must be >= {min_version}. Current version: {module.__version__}")
    except ImportError:
        logger.warning(f"{library} is not installed. Please install it to use interpreTS.")


# Available imports
from .core.feature_extractor import FeatureExtractor
from .utils.feature_loader import FeatureLoader, Features
from .utils.data_validation import validate_time_series_data
from .utils.data_manager import generate_feature_descriptions

__version__ = "0.4.1"

__all__ = [
    "FeatureExtractor",
    "Features",
    "FeatureLoader",
    "validate_time_series_data",
    "generate_feature_descriptions"
  #   "start_gui",
]