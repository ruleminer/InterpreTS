import pytest
import pandas as pd
import numpy as np
from interpreTS.core.aggregation import aggregate_time_series

def test_aggregate_time_series_valid_input():
    data = pd.DataFrame({
        'value': np.random.rand(100)
    }, index=pd.date_range('20210101', periods=100, freq='T'))
    
    features = [np.mean, np.std]
    result = aggregate_time_series(data, '1H', features)
    
    assert isinstance(result, pd.DataFrame)
    assert 'mean' in result.columns
    assert 'std' in result.columns
    assert len(result) == 2

def test_aggregate_time_series_invalid_data_type():
    data = [1, 2, 3, 4, 5]
    features = [np.mean, np.std]
    
    with pytest.raises(TypeError):
        aggregate_time_series(data, '1H', features)

def test_aggregate_time_series_invalid_freq_type():
    data = pd.DataFrame({
        'value': np.random.rand(100)
    }, index=pd.date_range('20210101', periods=100, freq='T'))
    
    features = [np.mean, np.std]
    
    with pytest.raises(TypeError):
        aggregate_time_series(data, 1, features)

def test_aggregate_time_series_invalid_features_type():
    data = pd.DataFrame({
        'value': np.random.rand(100)
    }, index=pd.date_range('20210101', periods=100, freq='T'))
    
    features = [np.mean, 'std']
    
    with pytest.raises(ValueError):
        aggregate_time_series(data, '1H', features)

def test_aggregate_time_series_invalid_index_type():
    data = pd.DataFrame({
        'value': np.random.rand(100)
    })
    
    features = [np.mean, np.std]
    
    with pytest.raises(ValueError):
        aggregate_time_series(data, '1H', features)