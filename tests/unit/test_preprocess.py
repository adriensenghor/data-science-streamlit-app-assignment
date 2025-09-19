"""
Tests for data preprocessing utilities.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.bike_counters.data.preprocess import (
    normalize_columns,
    clean_data,
    add_log_target,
    create_date_encoder,
    create_preprocessing_pipeline,
    create_full_pipeline,
    prepare_features,
    split_temporal_data
)


class TestDataPreprocessing:
    """Test cases for data preprocessing functions."""
    
    def test_normalize_columns(self):
        """Test column name normalization."""
        df = pd.DataFrame({
            '  Station ID  ': [1, 2],
            'TIMESTAMP': ['2023-01-01', '2023-01-02'],
            'Count': [10, 12]
        })
        
        normalized = normalize_columns(df)
        
        expected_columns = ['station id', 'timestamp', 'count']
        assert list(normalized.columns) == expected_columns
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create data with missing values and outliers
        df = pd.DataFrame({
            'bike_count': [10, 12, np.nan, -5, 1000],  # Missing, negative, outlier
            'date': pd.date_range('2023-01-01', periods=5, freq='H'),
            'counter_name': ['C1'] * 5
        })
        
        cleaned = clean_data(df)
        
        # Should remove missing values, negatives, and outliers
        assert len(cleaned) < len(df)
        assert cleaned['bike_count'].min() >= 0
        assert cleaned['bike_count'].max() <= df['bike_count'].quantile(0.99)
    
    def test_add_log_target(self):
        """Test log transformation of target variable."""
        df = pd.DataFrame({
            'bike_count': [0, 1, 10, 100]
        })
        
        result = add_log_target(df)
        
        expected_log = np.log1p(df['bike_count'])
        np.testing.assert_array_almost_equal(result['log_bike_count'], expected_log)
    
    def test_create_date_encoder(self):
        """Test date encoder creation and functionality."""
        encoder = create_date_encoder()
        
        # Test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01 12:00:00', periods=1, freq='H')
        })
        
        encoded = encoder.fit_transform(test_data)
        
        # Check that temporal features are created
        assert 'year' in encoded.columns
        assert 'month' in encoded.columns
        assert 'day' in encoded.columns
        assert 'weekday' in encoded.columns
        assert 'hour' in encoded.columns
        assert 'date' not in encoded.columns
        
        # Check values
        assert encoded['year'].iloc[0] == 2023
        assert encoded['month'].iloc[0] == 1
        assert encoded['hour'].iloc[0] == 12
    
    def test_create_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation."""
        pipeline = create_preprocessing_pipeline(
            categorical_cols=['counter_name'],
            numerical_cols=['latitude', 'longitude']
        )
        
        assert isinstance(pipeline, Pipeline)
        assert 'preprocessor' in pipeline.named_steps
    
    def test_prepare_features(self):
        """Test feature preparation."""
        df = pd.DataFrame({
            'counter_id': ['C1'],
            'counter_name': ['Test Counter'],
            'bike_count': [10],
            'log_bike_count': [2.3],
            'latitude': [48.8566],
            'longitude': [2.3522],
            'date': pd.date_range('2023-01-01', periods=1, freq='H')
        })
        
        features = prepare_features(df)
        
        # Should exclude target and specified columns
        assert 'log_bike_count' not in features.columns
        assert 'counter_id' not in features.columns
        assert 'bike_count' not in features.columns
        assert 'counter_name' in features.columns
        assert 'latitude' in features.columns
    
    def test_split_temporal_data_string(self):
        """Test temporal data splitting with string test size."""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        X = pd.DataFrame({
            'date': dates,
            'counter_name': ['C1'] * 100,
            'bike_count': np.random.randint(0, 50, 100)
        })
        y = pd.Series(np.random.randn(100))
        
        X_train, X_test, y_train, y_test = split_temporal_data(X, y, "10 hours")
        
        # Test set should be the last 10 hours
        assert len(X_test) == 10
        assert len(X_train) == 90
        assert X_test['date'].min() > X_train['date'].max()
    
    def test_split_temporal_data_fraction(self):
        """Test temporal data splitting with fraction test size."""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        X = pd.DataFrame({
            'date': dates,
            'counter_name': ['C1'] * 100,
            'bike_count': np.random.randint(0, 50, 100)
        })
        y = pd.Series(np.random.randn(100))
        
        X_train, X_test, y_train, y_test = split_temporal_data(X, y, 0.2)
        
        # Test set should be 20% of data
        assert len(X_test) == 20
        assert len(X_train) == 80
        assert X_test['date'].min() > X_train['date'].max()
