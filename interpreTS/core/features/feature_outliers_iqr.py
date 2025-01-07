import numpy as np
import pandas as pd

def calculate_outliers_iqr(data, training_data, epsilon=1e-6):
    """
    Calculates the percentage of observations in a given window that fall below (Q1 - 1.5 * IQR) 
    or above (Q3 + 1.5 * IQR) using the Interquartile Range (IQR) method.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        The data window to analyze for outliers.
    training_data : np.ndarray or pd.Series
        The training data used to calculate Q1 (25th percentile), Q3 (75th percentile), and IQR.
    epsilon : float, optional
        A small tolerance added to bounds when training data contains a single unique value 
        (default is 1e-6).

    Returns
    -------
    float
        The percentage of observations in the window that are considered outliers.

    Examples
    --------
    >>> import numpy as np
    >>> training_data = np.array([10, 12, 14, 15, 16, 18, 19])
    >>> data = np.array([9, 15, 20, 25])
    >>> calculate_outliers_iqr(data, training_data)
    0.25
    """
    if isinstance(training_data, pd.Series):
        training_data = training_data.values
    if isinstance(data, pd.Series):
        data = data.values

    # Handle single-value training data
    if np.all(training_data == training_data[0]):
        unique_value = training_data[0]
        lower_bound = unique_value - 1.5  # Adjusted for single value
        upper_bound = unique_value + 1.5
    else:
        # Calculate Q1, Q3, and IQR from the training dataset
        q1 = np.percentile(training_data, 25)
        q3 = np.percentile(training_data, 75)
        iqr = q3 - q1

        # Handle the case of zero IQR
        if iqr == 0:
            return 0.0

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

    # Count the number of outliers in the window
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    return outliers / len(data)
