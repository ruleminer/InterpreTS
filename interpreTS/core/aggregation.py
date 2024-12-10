import pandas as pd
import numpy as np

def aggregate_time_series(data, freq, features):
    """
    Aggregates a time series dataset using specified features and frequency.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with a datetime index and at least one column of values.
    freq : str
        Resampling frequency (e.g., '1H', '1D').
    features : list of callable
        List of functions to calculate features (e.g., mean, std, custom functions).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated features for each time period.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    if not isinstance(freq, str):
        raise TypeError("Frequency must be a string.")
    if not all(callable(f) for f in features):
        raise ValueError("Features must be a list of callable functions.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex.")

    def apply_features(window):
        return {func.__name__: func(window) for func in features}
    
    # Fix: Ensure `x` is treated as a Series and apply features directly
    aggregated = data.resample(freq).apply(lambda x: pd.Series(apply_features(x)))
    
    # Flatten multi-level index resulting from the `apply` and rename columns
    aggregated.columns = [col if isinstance(col, str) else f"feature_{i}" 
                          for i, col in enumerate(aggregated.columns)]
    return aggregated
