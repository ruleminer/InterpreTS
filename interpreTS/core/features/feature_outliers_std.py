import numpy as np
import pandas as pd

def calculate_outliers_std(data, training_data):
    """
    Calculates the percentage of observations in a window that are above or below 
    3 standard deviations from the mean, based on the training dataset.
    
    Parameters
    ----------
    data : np.ndarray or pd.Series
        Window data to analyze.
    training_data : np.ndarray or pd.Series
        Training data used to calculate the mean and standard deviation.

    Returns
    -------
    float
        Percentage of observations in the window that deviate by more than 3 standard deviations.
    """
    if isinstance(training_data, pd.Series):
        training_data = training_data.values
    if isinstance(data, pd.Series):
        data = data.values

    # Calculate mean and standard deviation from training data
    mean_value = np.mean(training_data)
    std_dev = np.std(training_data)

    # Define bounds for outliers (3 standard deviations from the mean)
    lower_bound = mean_value - 3 * std_dev
    upper_bound = mean_value + 3 * std_dev

    # Count observations outside the bounds
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    return outliers / len(data)
