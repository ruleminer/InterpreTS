import pandas as pd
import numpy as np

def validate_time_series_data(
    data,
    feature_name=None,
    validation_requirements=None,
    **kwargs
):
    """
    Validate the input time series data against dynamically provided requirements.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or np.ndarray
        The time series data to be validated.
    feature_name : str, optional
        The name of the feature to validate.
    validation_requirements : dict, optional
        A dictionary specifying the validation requirements for each feature.
    **kwargs : dict
        Additional validation parameters (overrides validation_requirements).

    Returns
    -------
    bool
        True if the data is valid; raises an error otherwise.

    Raises
    ------
    TypeError
        If data is not a pd.Series, pd.DataFrame, or np.ndarray.
    ValueError
        If any validation requirement is not met.
    """
    # Determine feature-specific requirements
    if feature_name and validation_requirements:
        requirements = validation_requirements.get(feature_name, {})
    else:
        requirements = kwargs

    # Check data type
    if not isinstance(data, (pd.Series, pd.DataFrame, np.ndarray)):
        raise TypeError("Data must be a pandas Series, DataFrame, or numpy array.")

    # Handle empty data
    if isinstance(data, (pd.Series, pd.DataFrame)) and data.empty:
        raise ValueError("Input data is empty.")
    elif isinstance(data, np.ndarray) and data.size == 0:
        raise ValueError("Input data is empty.")

    # Validate pandas Series or DataFrame
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # Check for NaN values
        if not requirements.get('allow_nan', True) and data.isnull().any().any():
            raise ValueError("Data contains NaN values.")
        
        # Check index type if datetime index is required
        if requirements.get('require_datetime_index', False) and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DateTime index for time-based operations.")
        
        # Check for numeric values
        if not np.issubdtype(data.to_numpy().dtype, np.number):
            raise TypeError("Data must contain only numeric values.")
    
    # Validate numpy array
    elif isinstance(data, np.ndarray):
        # Check if numeric
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("Data must contain only numeric values.")
        
        if data.ndim > 1 and not requirements.get('allow_nan', True):
            if np.isnan(data).any():
                raise ValueError("Data contains NaN values.")
    
    # Check for minimum length
    min_length = requirements.get('min_length', None)
    if min_length:
        if isinstance(min_length, str):  # Evaluate dynamic length expressions
            min_length = eval(min_length)
        if len(data) < min_length:
            raise ValueError(f"Data must have at least {min_length} points.")

    # Check for one-dimensional data
    if requirements.get('check_one_dimensional', False):
        if isinstance(data, np.ndarray) and data.ndim != 1:
            raise ValueError("Data must be one-dimensional.")
        if isinstance(data, pd.DataFrame) and data.shape[1] != 1:
            raise ValueError("Data must be one-dimensional.")

    # Additional validations
    if 'check_nonzero_mean' in requirements and requirements['check_nonzero_mean']:
        if np.isclose(data.mean(), 0):
            raise ValueError("Data mean must not be zero.")

    if 'validate_positive_parameters' in requirements:
        for param, error_message in requirements['validate_positive_parameters'].items():
            if param in kwargs and kwargs[param] <= 0:
                raise ValueError(error_message)

    if 'positive_integer_params' in requirements:
        for param in requirements['positive_integer_params']:
            if param in kwargs and (not isinstance(kwargs[param], int) or kwargs[param] <= 0):
                raise ValueError(f"{param} must be a positive integer.")

    return True
