import numpy as np

def calculate_outliers_iqr(data, training_data):
    """
    Oblicza procent obserwacji w oknie, które są poniżej (Q1 - 1.5 * IQR) lub powyżej (Q3 + 1.5 * IQR).
    
    Parameters
    ----------
    data : np.ndarray or pd.Series
        Dane z okna do analizy.
    training_data : np.ndarray or pd.Series
        Dane treningowe do obliczenia Q1, Q3 i IQR.

    Returns
    -------
    float
        Procent obserwacji w oknie odbiegających od wyznaczonych granic.
    """
    if isinstance(training_data, pd.Series):
        training_data = training_data.values
    if isinstance(data, pd.Series):
        data = data.values

    # Obliczenie Q1, Q3 i IQR na podstawie zbioru treningowego
    q1 = np.percentile(training_data, 25)
    q3 = np.percentile(training_data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Liczenie obserwacji poza granicami
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    return outliers / len(data)
