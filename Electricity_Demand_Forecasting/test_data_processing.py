import pytest
import pandas as pd
from src.data_processing import generate_synthetic_data, load_and_preprocess_data

def test_synthetic_data_generation():
    df = generate_synthetic_data(days=10)
    assert not df.empty
    assert 'timestamp' in df.columns
    assert 'demand' in df.columns
    assert len(df) == 10 * 24

def test_data_preprocessing_synthetic():
    df_raw = generate_synthetic_data(days=2)
    # Test passing dataframe directly or via mock
    df_clean = load_and_preprocess_data(df_raw)
    assert not df_clean.empty
    assert df_clean['timestamp'].is_monotonic_increasing
    assert df_clean['demand'].isnull().sum() == 0