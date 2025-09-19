"""
Tests for feature engineering utilities.
"""

import pytest
import pandas as pd
import numpy as np

from src.bike_counters.features.engineering import (
    encode_dates,
    add_hour,
    add_cyclical_features,
    add_weather_features,
    add_location_features,
    create_lag_features,
    create_rolling_features
)


class TestFeatureEngineering:
    """Test cases for feature engineering functions."""
    
    def test_encode_dates(self):
        """Test date encoding functionality."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01 12:30:00', periods=2, freq='D')
        })
        
        result = encode_dates(df)
        
        # Check temporal features
        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 'day' in result.columns
        assert 'weekday' in result.columns
        assert 'hour' in result.columns
        assert 'day_of_year' in result.columns
        assert 'week_of_year' in result.columns
        
        # Check cyclical features
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'weekday_sin' in result.columns
        assert 'weekday_cos' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        
        # Check values
        assert result['year'].iloc[0] == 2023
        assert result['month'].iloc[0] == 1
        assert result['hour'].iloc[0] == 12
    
    def test_add_hour(self):
        """Test hour feature addition."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 14:30:00', periods=2, freq='H')
        })
        
        result = add_hour(df)
        
        assert 'hour' in result.columns
        assert result['hour'].iloc[0] == 14
        assert result['hour'].iloc[1] == 15
    
    def test_add_cyclical_features(self):
        """Test cyclical feature encoding."""
        df = pd.DataFrame({
            'hour': [0, 6, 12, 18],
            'weekday': [0, 1, 2, 3],
            'month': [1, 4, 7, 10]
        })
        
        result = add_cyclical_features(df)
        
        # Check cyclical features are created
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'weekday_sin' in result.columns
        assert 'weekday_cos' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        
        # Check cyclical properties
        assert np.allclose(result['hour_sin']**2 + result['hour_cos']**2, 1.0)
    
    def test_add_weather_features(self):
        """Test weather feature addition."""
        df = pd.DataFrame({
            'counter_name': ['C1', 'C2'],
            'date': pd.date_range('2023-01-01', periods=2, freq='H')
        })
        
        result = add_weather_features(df)
        
        assert 'temperature' in result.columns
        assert 'precipitation' in result.columns
        assert 'wind_speed' in result.columns
        
        # Check default values
        assert result['temperature'].iloc[0] == 15.0
        assert result['precipitation'].iloc[0] == 0.0
        assert result['wind_speed'].iloc[0] == 5.0
    
    def test_add_location_features(self):
        """Test location feature addition."""
        df = pd.DataFrame({
            'latitude': [48.8566, 48.8600],
            'longitude': [2.3522, 2.3600]
        })
        
        result = add_location_features(df)
        
        assert 'distance_from_center' in result.columns
        assert 'zone' in result.columns
        
        # Check distance calculation
        assert result['distance_from_center'].iloc[0] < result['distance_from_center'].iloc[1]
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        df = pd.DataFrame({
            'counter_name': ['C1'] * 10,
            'date': pd.date_range('2023-01-01', periods=10, freq='H'),
            'bike_count': range(10)
        })
        
        result = create_lag_features(df, lags=[1, 2])
        
        assert 'bike_count_lag_1' in result.columns
        assert 'bike_count_lag_2' in result.columns
        
        # Check lag values
        assert result['bike_count_lag_1'].iloc[2] == 1  # Should be bike_count at index 1
        assert result['bike_count_lag_2'].iloc[2] == 0  # Should be bike_count at index 0
    
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        df = pd.DataFrame({
            'counter_name': ['C1'] * 10,
            'date': pd.date_range('2023-01-01', periods=10, freq='H'),
            'bike_count': range(10)
        })
        
        result = create_rolling_features(df, windows=[3])
        
        assert 'bike_count_rolling_mean_3' in result.columns
        assert 'bike_count_rolling_std_3' in result.columns
        
        # Check rolling calculations
        assert not pd.isna(result['bike_count_rolling_mean_3'].iloc[2])
        assert not pd.isna(result['bike_count_rolling_std_3'].iloc[2])
