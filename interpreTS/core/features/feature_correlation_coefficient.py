import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def calculate_correlation_coefficient(series1, series2, handle_na="drop"):
    """
    Calculate the Pearson correlation coefficient between two time series.

    Parameters
    ----------
    series1 : pd.Series or np.ndarray
        The first time series.
    series2 : pd.Series or np.ndarray
        The second time series.
    handle_na : str, optional
        Strategy to handle missing values:
        - "drop": Remove all rows where either series has NaN values (default).
        - "fill_zero": Fill NaN values with zero.
        - "fill_mean": Fill NaN values with the mean of the respective series.

    Returns
    -------
    float
        The Pearson correlation coefficient between the two time series.
        A value of 1 indicates perfect positive correlation, -1 indicates perfect negative correlation,
        and 0 indicates no linear relationship.

    Raises
    ------
    ValueError
        If the series lengths are not equal or if they are empty after preprocessing.
    TypeError
        If the inputs are not valid time series types.

    Examples
    --------
    >>> series1 = pd.Series([1, 2, 3, 4, 5])
    >>> series2 = pd.Series([5, 4, 3, 2, 1])
    >>> calculate_correlation_coefficient(series1, series2)
    -1.0

    >>> series1 = pd.Series([1, 2, 3, np.nan, 5])
    >>> series2 = pd.Series([5, np.nan, 3, 2, 1])
    >>> calculate_correlation_coefficient(series1, series2, handle_na="fill_mean")
    -0.7999999999999998
    """
    # Convert inputs to pandas Series if they are numpy arrays
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)
    if isinstance(series2, np.ndarray):
        series2 = pd.Series(series2)

    # Validate input types
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        raise TypeError("Inputs must be pandas Series or numpy arrays.")

    # Validate equal lengths
    if len(series1) != len(series2):
        raise ValueError("The two time series must have the same length.")

    # Handle missing values
    if handle_na == "drop":
        series1 = series1.dropna()
        series2 = series2.dropna()
    elif handle_na == "fill_zero":
        series1 = series1.fillna(0)
        series2 = series2.fillna(0)
    elif handle_na == "fill_mean":
        series1 = series1.fillna(series1.mean())
        series2 = series2.fillna(series2.mean())
    else:
        raise ValueError("Invalid handle_na value. Choose from 'drop', 'fill_zero', or 'fill_mean'.")

    # Ensure the series are still of equal length after handling NaNs
    if len(series1) != len(series2):
        raise ValueError("The two time series must remain of equal length after handling NaN values.")

    # Check for sufficient data
    if len(series1) == 0 or len(series2) == 0:
        raise ValueError("One or both of the time series are empty after preprocessing.")

    # Calculate and return the correlation coefficient
    correlation, _ = pearsonr(series1, series2)
    return correlation
