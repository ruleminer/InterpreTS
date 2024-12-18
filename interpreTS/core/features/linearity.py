import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_linearity(data, normalize=True, use_derivative=True):
    """
    Calculate the linearity of a time series.
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which the linearity is to be calculated.
    normalize : bool, optional
        Whether to normalize the data before calculating linearity (default is True).
    use_derivative : bool, optional
        Whether to calculate linearity on the first derivative of the data (default is True).
    Returns
    -------
    float
        The R-squared value representing the linearity of the time series.
        A value closer to 1 indicates higher linearity.
    Raises
    ------
    TypeError
        If the data is not a valid time series type or contains non-numeric values.
    ValueError
        If the data is empty or contains insufficient unique points for regression.
    """
    # Validate the time series data
    validate_time_series_data(data, require_datetime_index=False)

    # Convert data to pandas Series if it's an ndarray
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Drop NaN values
    data = data.dropna()

    # Handle empty series
    if len(data) == 0:
        raise ValueError("The time series is empty after removing NaN values.")

    # Ensure data is numeric
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must contain only numeric values.")

    # Optionally normalize the data
    if normalize:
        data = (data - data.mean()) / data.std()

    # Optionally calculate the first derivative (differences)
    if use_derivative:
        data = data.diff().dropna()

    # Generate a range of indices as the independent variable
    x = np.arange(len(data)).reshape(-1, 1)
    y = data.values

    # Handle cases with insufficient unique points for regression
    if len(np.unique(y)) < 2:
        return 0.0  # Return 0.0 for constant or near-constant series

    # Perform linear regression
    model = LinearRegression()
    model.fit(x, y)
    r_squared = model.score(x, y)

    return r_squared

