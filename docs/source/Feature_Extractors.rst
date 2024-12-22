Feature Extractors
==================

This section provides an overview of the feature extractor and the features available in the library.

## Feature Extractor

The `Feature Extractor` is the core component that extracts meaningful metrics from time-series data.

.. automodule:: interpreTS.core.feature_extractor
   :members:
   :undoc-members:
   :show-inheritance:

---

Available Features
==================

Below is a list of the available features in the library. Each feature is automatically documented from the code, with a brief description.

Length
==================
Extracts the total length of a time series.

.. automodule:: interpreTS.core.features.feature_length
   :members:
   :undoc-members:
   :show-inheritance:

Mean
==================
Calculates the mean value of a time series.

.. automodule:: interpreTS.core.features.feature_mean
   :members:
   :undoc-members:
   :show-inheritance:

Peak
==================
Identifies the maximum peak value.

.. automodule:: interpreTS.core.features.feature_peak
   :members:
   :undoc-members:
   :show-inheritance:

Spikeness
==================
Measures the level of spikeness in the time series.

.. automodule:: interpreTS.core.features.feature_spikeness
   :members:
   :undoc-members:
   :show-inheritance:

Standard Deviation of the First Derivative (Std_1st_der)
==================
Calculates the standard deviation of the first derivative of the series.

.. automodule:: interpreTS.core.features.feature_std_1st_der
   :members:
   :undoc-members:
   :show-inheritance:

Trough
==================
Identifies the lowest point in the time series.

.. automodule:: interpreTS.core.features.feature_trough
   :members:
   :undoc-members:
   :show-inheritance:

Variance
==================
Computes the variance of the series.

.. automodule:: interpreTS.core.features.feature_variance
   :members:
   :undoc-members:
   :show-inheritance:

Dominant
==================
Finds the most dominant value in a histogram representation.

.. automodule:: interpreTS.core.features.histogram_dominant
   :members:
   :undoc-members:
   :show-inheritance:

Seasonality Strength
==================
Assesses the strength of seasonality patterns in the data.

.. automodule:: interpreTS.core.features.seasonality_strength
   :members:
   :undoc-members:
   :show-inheritance:

Trend Strength
==================
Measures the strength of the overall trend in the series.

.. automodule:: interpreTS.core.features.trend_strength
   :members:
   :undoc-members:
   :show-inheritance:

---

Notes
==================

Each feature is designed to provide specific insights into time-series data. For detailed usage, refer to the module documentation linked above.
