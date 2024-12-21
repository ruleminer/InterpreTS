Feature Extractors
==================

This library provides feature extractors to process time series data. Each extractor computes specific characteristics.

Available Features:
-------------------
.. list-table:: Key Feature Extractors
   :header-rows: 1

   * - Extractor
     - Description
     - Example
   * - `mean_feature`
     - Computes the average of a time series.
     - `mean_feature([1, 2, 3]) -> 2.0`
   * - `variance_feature`
     - Computes the variance of a time series.
     - `variance_feature([1, 2, 3]) -> 0.67`
   * - `seasonality_strength`
     - Measures seasonality in a time series.
     - `seasonality_strength(data) -> 0.8`

Code Example:
-------------
.. code-block:: python

   from interpreTS.core.features import mean_feature

   series = [10, 20, 30]
   mean = mean_feature(series)
   print(f"Mean value: {mean}")
