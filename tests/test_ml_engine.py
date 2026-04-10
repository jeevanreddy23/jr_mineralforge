"""
Tests – ML Engine Agent
Tests for spatial CV, outlier detection, and ensemble training.
"""

import pytest
import pandas as pd
import numpy as np
from agents.ml_engine_agent import (
    SpatialBlockCV,
    engineer_geophysical_features,
    remove_outliers_isolation_forest,
    ProspectivityMLEngine
)

@pytest.fixture
def mock_geophysics_data():
    """Create a mock geophysics dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'tmi': np.random.normal(0, 100, n),
        'gravity': np.random.normal(0, 10, n),
        'k_pct': np.random.uniform(0, 5, n),
        'eth': np.random.uniform(0, 10, n),
        'eu': np.random.uniform(0, 5, n),
        'label': np.random.choice([0, 1], n, p=[0.9, 0.1])
    })
    
    # Coordinates in UTM (metres)
    coords = pd.DataFrame({
        'x': np.linspace(500000, 510000, n),
        'y': np.linspace(6700000, 6710000, n)
    })
    return df, coords

def test_feature_engineering(mock_geophysics_data):
    df, _ = mock_geophysics_data
    engineered_df = engineer_geophysical_features(df)
    
    assert 'k_eth_ratio' in engineered_df.columns
    assert 'f_parameter' in engineered_df.columns
    assert 'tmi_z_score' in engineered_df.columns

def test_outlier_removal(mock_geophysics_data):
    df, _ = mock_geophysics_data
    feat_cols = [c for c in df.columns if c != 'label']
    X = df[feat_cols]
    
    cleaned_X, mask = remove_outliers_isolation_forest(X, contamination=0.1)
    
    assert len(cleaned_X) < len(X)
    assert len(mask) == len(X)
    assert mask.sum() == len(cleaned_X)

def test_spatial_cv_split(mock_geophysics_data):
    df, coords = mock_geophysics_data
    cv = SpatialBlockCV(n_folds=5, buffer_km=1.0)
    
    splits = list(cv.split(df, df['label'], coords))
    
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Ensure no overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0

def test_ml_engine_validate_data(mock_geophysics_data):
    df, _ = mock_geophysics_data
    # Add some NaNs
    df.iloc[0, 0] = np.nan
    
    engine = ProspectivityMLEngine()
    validated_df = engine.validate_data(df)
    
    assert validated_df.isnull().sum().sum() == 0
    assert len(validated_df) <= len(df)
