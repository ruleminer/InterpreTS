Feature Extractors
==================

This section provides an overview of the feature extractor and the features available in the library.

The `Feature Extractor` is the core component that extracts meaningful metrics and features from time-series data.

.. automodule:: interpreTS.core.feature_extractor
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:


Available Features
==================

Below is a list of the available features in the library. Each feature is automatically documented from the code, with a brief description.

Length
------
Extracts the total length of a time series.

.. automodule:: interpreTS.core.features.feature_length
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Mean
----
Calculates the mean value of a time series.

.. automodule:: interpreTS.core.features.feature_mean
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Peak
----
Identifies the maximum peak value.

.. automodule:: interpreTS.core.features.feature_peak
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Spikeness
---------
Measures the level of spikeness in the time series.

.. automodule:: interpreTS.core.features.feature_spikeness
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Standard Deviation of the First Derivative (Std_1st_der)
--------------------------------------------------------
Calculates the standard deviation of the first derivative of the series.

.. automodule:: interpreTS.core.features.feature_std_1st_der
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Trough
------
Identifies the lowest point in the time series.

.. automodule:: interpreTS.core.features.feature_trough
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Variance
--------
Computes the variance of the series.

.. automodule:: interpreTS.core.features.feature_variance
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Dominant
--------
Finds the most dominant value in a histogram representation.

.. automodule:: interpreTS.core.features.histogram_dominant
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Seasonality Strength
--------------------
Assesses the strength of seasonality patterns in the data.

.. automodule:: interpreTS.core.features.seasonality_strength
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Trend Strength
--------------
Measures the strength of the overall trend in the series.

.. automodule:: interpreTS.core.features.trend_strength
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Above 9th Decile
----------------
Calculates whether the values in the time series are above the 9th decile.

.. automodule:: interpreTS.core.features.feature_above_9th_decile
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Distance to the Last Change Point
---------------------------------
Measures the distance to the last change point in the time series.

.. automodule:: interpreTS.core.features.distance_to_the_last_change_point
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Absolute Energy
---------------
Calculates the absolute energy of the time series.

.. automodule:: interpreTS.core.features.feature_absolute_energy
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Below 1st Decile
----------------
Calculates whether the values in the time series are below the 1st decile.

.. automodule:: interpreTS.core.features.feature_below_1st_decile
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Binarize Mean
-------------
Binarizes the mean value of the time series.

.. automodule:: interpreTS.core.features.feature_binarize_mean
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Crossing Points
---------------
Counts the crossing points in the time series.

.. automodule:: interpreTS.core.features.feature_crossing_points
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Entropy
-------
Calculates the entropy of the time series.

.. automodule:: interpreTS.core.features.feature_entropy
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Flat Spots
----------
Identifies flat spots within the time series.

.. automodule:: interpreTS.core.features.feature_flat_spots
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Heterogeneity
-------------
Measures the heterogeneity of the time series.

.. automodule:: interpreTS.core.features.feature_heterogeneity
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Linearity
---------
Calculates the linearity of the time series.

.. automodule:: interpreTS.core.features.feature_linearity
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Missing Points
--------------
Identifies missing points within the time series.

.. automodule:: interpreTS.core.features.feature_missing_points
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Outliers IQR
------------
Identifies outliers based on the interquartile range (IQR).

.. automodule:: interpreTS.core.features.feature_outliers_iqr
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Outliers STD
------------
Identifies outliers based on standard deviation (STD).

.. automodule:: interpreTS.core.features.feature_outliers_std
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Significant Changes
-------------------
Detects significant changes in the time series.

.. automodule:: interpreTS.core.features.feature_significant_changes
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Stability
---------
Measures the stability of the time series.

.. automodule:: interpreTS.core.features.feature_stability
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Variance Change
---------------
Calculates the variance change over time.

.. automodule:: interpreTS.core.features.variance_change
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Variability in Sub-Periods
--------------------------
Measures variability within sub-periods of the time series.

.. automodule:: interpreTS.core.features.variability_in_sub_periods
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Amplitude Change Rate
---------------------
Calculates the rate of amplitude change in the time series.

.. automodule:: interpreTS.core.features.feature_amplitude_change_rate
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:


Notes
-----
Each feature is designed to provide specific insights into time-series data. For detailed usage, refer to the module documentation linked above.
