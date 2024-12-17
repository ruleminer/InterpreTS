import pytest
import pandas as pd
import numpy as np
from interpreTS.core.features.feature_significant_changes import calculate_significant_changes

# Test 1: Standardowy przypadek z różnymi wartościami
def test_significant_changes_standard_case():
    """Test dla standardowego przypadku z różnymi wartościami."""
    data = pd.Series([1, 2, 1.5, 3, 2.5, 5, 4.5])
    result = calculate_significant_changes(data)
    expected = 0.3333333333333333  # 2 znaczące zmiany na 6 różnic
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test 2: Dane jednopunktowe
def test_significant_changes_single_point():
    """Test dla danych zawierających tylko jeden punkt."""
    data = pd.Series([1])
    result = calculate_significant_changes(data)
    expected = 0.0  # Nie można obliczyć różnic
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test 3: Dane o stałych wartościach
def test_significant_changes_constant_values():
    """Test dla stałych wartości (brak znaczących zmian)."""
    data = pd.Series([5, 5, 5, 5, 5])
    result = calculate_significant_changes(data)
    expected = 0.0  # Brak różnic, więc brak znaczących zmian
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test 4: Dane zawierające NaN
def test_significant_changes_with_nan():
    """Test dla danych zawierających NaN."""
    data = pd.Series([1, 2, np.nan, 4, 5])
    with pytest.raises(ValueError, match="Data contains NaN values."):
        calculate_significant_changes(data)

# Test 5: Dane o długości 2
def test_significant_changes_two_points():
    """Test dla danych zawierających tylko 2 punkty."""
    data = pd.Series([1, 2])
    result = calculate_significant_changes(data)
    expected = 0.0  # Dla 1 różnicy, nie można ocenić "znaczących zmian"
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test 6: Przypadek z ujemnymi wartościami
def test_significant_changes_negative_values():
    """Test dla danych zawierających ujemne wartości."""
    data = pd.Series([-5, -10, -5, -15, -5, -20])
    result = calculate_significant_changes(data)
    assert result > 0.0, f"Expected a positive proportion, got {result}"

# Test 7: Przypadek z pustymi danymi
def test_significant_changes_empty_data():
    """Test dla pustego szeregu."""
    data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input data is empty."):
        calculate_significant_changes(data)

# Test 8: Dane jako numpy array
def test_significant_changes_numpy_array():
    """Test dla danych w formacie numpy array."""
    data = np.array([1, 2, 1.5, 3, 2.5, 5, 4.5])
    result = calculate_significant_changes(data)
    expected = 0.3333333333333333
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test 9: Dane monotoniczne rosnące
def test_significant_changes_monotonic_increasing():
    """Test dla danych monotonicznie rosnących."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_significant_changes(data)
    expected = 0.0  # Brak znaczących zmian
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"

# Test 10: Dane monotoniczne malejące
def test_significant_changes_monotonic_decreasing():
    """Test dla danych monotonicznie malejących."""
    data = pd.Series([5, 4, 3, 2, 1])
    result = calculate_significant_changes(data)
    expected = 0.0  # Brak znaczących zmian
    assert result == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {result}"
