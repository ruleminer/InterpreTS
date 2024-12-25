API Documentation
==================

Functions
---------

Convert to Data Series Functions
--------------------------------
This module provides utilities for converting various data formats into time-series compatible structures. These utilities ensure the data is properly formatted and ready for analysis.

.. automodule:: interpreTS.utils.data_conversion
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Data Manager
------------
This module handles operations related to managing metadata, feature functions, and validation requirements necessary for extracting features from time-series data.

.. automodule:: interpreTS.utils.data_manager
   :members:
   :undoc-members:
   :show-inheritance:

Data Validation
---------------
This module provides comprehensive functions to ensure that time-series data adheres to the expected format, dimensionality, and integrity. It includes checks for missing values, data type consistency, and other preprocessing requirements.

.. automodule:: interpreTS.utils.data_validation
   :members:
   :undoc-members:
   :show-inheritance:

Features Loader
---------------
This module enables dynamic loading and management of feature extraction functions available in the library. It provides an interface for accessing and utilizing predefined features.

.. automodule:: interpreTS.utils.feature_loader
   :members:
   :undoc-members:
   :show-inheritance:

Task Manager
------------
This module is responsible for orchestrating feature extraction tasks. It includes task generation, validation, execution, and parallelization using various computational backends.

.. automodule:: interpreTS.utils.task_manager
   :members:
   :undoc-members:
   :show-inheritance:
