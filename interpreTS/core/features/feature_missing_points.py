import pandas as pd
import numpy as np

def calculate_missing_points(data, percentage=True):
    """
    Calculate the percentage or count of missing (NaN or None) values in a time series.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which missing information is to be calculated.
    percentage : bool, optional
        If True, returns the percentage of missing values.
        If False, returns the count of missing values.
        Default is True.

    Returns
    -------
    float or int
        The percentage or count of missing values in the provided time series.

    Raises
    ------
    TypeError
        If the data is not a valid time series type.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, np.nan, 4, None])
    >>> missing_points(data)
    0.4
    >>> missing_points(data, percentage=False)
    2
    """

    # Convert to a pandas Series if data is an ndarray
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Calculate total and missing values
    total_values = len(data)
    missing_values = data.isna().sum()

    # Return either percentage or count of missing values
    if total_values == 0:
        return np.nan
    return missing_values / total_values if percentage else missing_values