Feature Extractors
==================

This section provides an overview of the feature extractor and the features available in the library.

The `Feature Extractor` is the core component that extracts meaningful metrics and features from time-series data.

.. automodule:: interpreTS.core.feature_extractor
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
   :members:
   :undoc-members:
   :show-inheritance:

Mean
----
Calculates the mean value of a time series.

.. automodule:: interpreTS.core.features.feature_mean
   :members:
   :undoc-members:
   :show-inheritance:

Peak
----
Identifies the maximum peak value.

.. automodule:: interpreTS.core.features.feature_peak
   :members:
   :undoc-members:
   :show-inheritance:

Spikeness
---------
Measures the level of spikeness in the time series.

.. automodule:: interpreTS.core.features.feature_spikeness
   :members:
   :undoc-members:
   :show-inheritance:

Standard Deviation of the First Derivative (Std_1st_der)
--------------------------------------------------------
Calculates the standard deviation of the first derivative of the series.

.. automodule:: interpreTS.core.features.feature_std_1st_der
   :members:
   :undoc-members:
   :show-inheritance:

Trough
------
Identifies the lowest point in the time series.

.. automodule:: interpreTS.core.features.feature_trough
   :members:
   :undoc-members:
   :show-inheritance:

Variance
--------
Computes the variance of the series.

.. automodule:: interpreTS.core.features.feature_variance
   :members:
   :undoc-members:
   :show-inheritance:

Dominant
--------
Finds the most dominant value in a histogram representation.

.. automodule:: interpreTS.core.features.feature_histogram_dominant
   :members:
   :undoc-members:
   :show-inheritance:

Seasonality Strength
--------------------
Assesses the strength of seasonality patterns in the data.

.. automodule:: interpreTS.core.features.feature_seasonality_strength
   :members:
   :undoc-members:
   :show-inheritance:

Trend Strength
--------------
Measures the strength of the overall trend in the series.

.. automodule:: interpreTS.core.features.feature_trend_strength
   :members:
   :undoc-members:
   :show-inheritance:

Above 9th Decile
----------------
Calculates whether the values in the time series are above the 9th decile.

.. automodule:: interpreTS.core.features.feature_above_9th_decile
   :members:
   :undoc-members:
   :show-inheritance:

Distance to the Last Change Point
---------------------------------
Measures the distance to the last change point in the time series.

.. automodule:: interpreTS.core.features.feature_distance_to_the_last_change_point
   :members:
   :undoc-members:
   :show-inheritance:

Absolute Energy
---------------
Calculates the absolute energy of the time series.

.. automodule:: interpreTS.core.features.feature_absolute_energy
   :members:
   :undoc-members:
   :show-inheritance:

Below 1st Decile
----------------
Calculates whether the values in the time series are below the 1st decile.

.. automodule:: interpreTS.core.features.feature_below_1st_decile
   :members:
   :undoc-members:
   :show-inheritance:

Binarize Mean
-------------
Binarizes the mean value of the time series.

.. automodule:: interpreTS.core.features.feature_binarize_mean
   :members:
   :undoc-members:
   :show-inheritance:

Crossing Points
---------------
Counts the crossing points in the time series.

.. automodule:: interpreTS.core.features.feature_crossing_points
   :members:
   :undoc-members:
   :show-inheritance:

Entropy
-------
Calculates the entropy of the time series.

.. automodule:: interpreTS.core.features.feature_entropy
   :members:
   :undoc-members:
   :show-inheritance:

Flat Spots
----------
Identifies flat spots within the time series.

.. automodule:: interpreTS.core.features.feature_flat_spots
   :members:
   :undoc-members:
   :show-inheritance:

Heterogeneity
-------------
Measures the heterogeneity of the time series.

.. automodule:: interpreTS.core.features.feature_heterogeneity
   :members:
   :undoc-members:
   :show-inheritance:

Linearity
---------
Calculates the linearity of the time series.

.. automodule:: interpreTS.core.features.feature_linearity
   :members:
   :undoc-members:
   :show-inheritance:

Missing Points
--------------
Identifies missing points within the time series.

.. automodule:: interpreTS.core.features.feature_missing_points
   :members:
   :undoc-members:
   :show-inheritance:

Outliers IQR
------------
Identifies outliers based on the interquartile range (IQR).

.. automodule:: interpreTS.core.features.feature_outliers_iqr
   :members:
   :undoc-members:
   :show-inheritance:

Outliers STD
------------
Identifies outliers based on standard deviation (STD).

.. automodule:: interpreTS.core.features.feature_outliers_std
   :members:
   :undoc-members:
   :show-inheritance:

Significant Changes
-------------------
Detects significant changes in the time series.

.. automodule:: interpreTS.core.features.feature_significant_changes
   :members:
   :undoc-members:
   :show-inheritance:

Stability
---------
Measures the stability of the time series.

.. automodule:: interpreTS.core.features.feature_stability
   :members:
   :undoc-members:
   :show-inheritance:

Variance Change
---------------
Calculates the variance change over time.

.. automodule:: interpreTS.core.features.feature_variance_change
   :members:
   :undoc-members:
   :show-inheritance:

Variability in Sub-Periods
--------------------------
Measures variability within sub-periods of the time series.

.. automodule:: interpreTS.core.features.feature_variability_in_sub_periods
   :members:
   :undoc-members:
   :show-inheritance:

Amplitude Change Rate
---------------------
Calculates the rate of amplitude change in the time series.

.. automodule:: interpreTS.core.features.feature_amplitude_change_rate
   :members:
   :undoc-members:
   :show-inheritance:


Notes
-----
Each feature is designed to provide specific insights into time-series data. For detailed usage, refer to the module documentation linked above.
