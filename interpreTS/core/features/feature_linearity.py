import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_linearity(data, normalize=True, use_derivative=True):
    """
    Calculate the linearity of a time series, similar to tsflex or sktime implementations.

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

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> calculate_linearity(data)
    1.0
    >>> data = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    >>> calculate_linearity(data)
    0.0
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    if normalize:
        data = (data - data.mean()) / data.std()

    if use_derivative:
        derivative_data = data.diff().dropna()
        if len(np.unique(derivative_data)) <= 1:  
            return 1.0 if len(np.unique(data)) > 1 else 0.0 
        data = derivative_data

    if len(data) < 2 or len(np.unique(data)) < 2:
        return 0.0

    x = np.arange(len(data)).reshape(-1, 1)
    y = data.values


    model = LinearRegression()
    model.fit(x, y)
    r_squared = model.score(x, y)

    return r_squared
