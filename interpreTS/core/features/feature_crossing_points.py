import numpy as np
import pandas as pd
from interpreTS.utils.data_validation import validate_time_series_data

def calculate_crossing_points(data):
    """
    Calculate the number of times and the list of indices where the time series crosses its mean.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which mean crossings are to be calculated.
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'crossing_count': The total number of crossings.
        - 'crossing_points': A list of indices where crossings occur.
    
    Raises
    ------
    ValueError
        If the data contains NaN values or is empty.
    """
    # Walidacja danych
    validate_time_series_data(data)

    # Konwersja do np.ndarray, jeśli to seria pandas
    if isinstance(data, pd.Series):
        data = data.values

    # Sprawdzenie, czy dane są puste
    if len(data) == 0:
        return {'crossing_count': 0, 'crossing_points': []}

    mean_value = np.mean(data)

    # Obliczenie powyżej/poniżej średniej
    above_mean = data > mean_value

    # Sprawdzenie, czy wszystkie wartości są powyżej lub poniżej średniej
    if np.all(above_mean) or np.all(~above_mean):  # Wszystkie powyżej lub wszystkie poniżej
        return {'crossing_count': 0, 'crossing_points': []}

    # Identyfikacja przecięć
    crossings = np.where(np.diff(above_mean.astype(int)) != 0)[0] + 1

    # Dodatkowe sprawdzenie: upewnij się, że wynik jest poprawny
    crossing_count = len(crossings) if len(crossings) > 0 else 0
    crossing_points = list(crossings) if len(crossings) > 0 else []

    return {
        'crossing_count': crossing_count,
        'crossing_points': crossing_points
    }