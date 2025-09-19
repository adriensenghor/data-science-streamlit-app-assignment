"""
Tests for data loading utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile
import os

from src.bike_counters.data.loader import (
    load_csv,
    load_parquet,
    load_train_data,
    load_test_data,
    validate_data_schema
)


class TestDataLoader:
    """Test cases for data loading functions."""
    
    def test_load_csv(self, tmp_path):
        """Test CSV loading functionality."""
        # Create test CSV
        test_data = pd.DataFrame({
            'station_id': [1, 2],
            'timestamp': ['2023-01-01T00:00:00', '2023-01-01T01:00:00'],
            'count': [10, 12]
        })
        
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Test loading
        loaded_data = load_csv(csv_path)
        
        assert len(loaded_data) == 2
        assert list(loaded_data.columns) == ['station_id', 'timestamp', 'count']
    
    def test_load_parquet(self, tmp_path):
        """Test Parquet loading functionality."""
        # Create test data
        test_data = pd.DataFrame({
            'counter_id': ['C1', 'C2'],
            'date': pd.date_range('2023-01-01', periods=2, freq='H'),
            'bike_count': [10, 12]
        })
        
        parquet_path = tmp_path / "test.parquet"
        test_data.to_parquet(parquet_path, index=False)
        
        # Test loading
        loaded_data = load_parquet(parquet_path)
        
        assert len(loaded_data) == 2
        assert 'counter_id' in loaded_data.columns
        assert 'bike_count' in loaded_data.columns
    
    def test_load_train_data_file_not_found(self):
        """Test error handling when train data file is missing."""
        with pytest.raises(FileNotFoundError):
            load_train_data("nonexistent_dir")
    
    @patch('src.bike_counters.data.loader.pd.read_parquet')
    def test_load_train_data_success(self, mock_read_parquet):
        """Test successful train data loading."""
        # Mock data
        mock_data = pd.DataFrame({
            'counter_id': ['C1'],
            'date': pd.date_range('2023-01-01', periods=1, freq='H'),
            'bike_count': [10],
            'log_bike_count': [2.3]
        })
        mock_read_parquet.return_value = mock_data
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            X, y = load_train_data()
            
            assert len(X) == 1
            assert len(y) == 1
            assert 'log_bike_count' not in X.columns
            assert y.iloc[0] == 2.3
    
    def test_load_test_data_file_not_found(self):
        """Test error handling when test data file is missing."""
        with pytest.raises(FileNotFoundError):
            load_test_data("nonexistent_dir")
    
    def test_validate_data_schema_success(self):
        """Test successful schema validation."""
        valid_data = pd.DataFrame({
            'counter_id': ['C1'],
            'counter_name': ['Test Counter'],
            'site_id': [1],
            'site_name': ['Test Site'],
            'bike_count': [10],
            'date': pd.date_range('2023-01-01', periods=1, freq='H'),
            'counter_installation_date': pd.date_range('2023-01-01', periods=1, freq='H'),
            'coordinates': ['48.8566,2.3522'],
            'counter_technical_id': ['T1'],
            'latitude': [48.8566],
            'longitude': [2.3522]
        })
        
        # Should not raise any exception
        validate_data_schema(valid_data)
    
    def test_validate_data_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        invalid_data = pd.DataFrame({
            'counter_id': ['C1'],
            'bike_count': [10]
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data_schema(invalid_data)
    
    def test_validate_data_schema_wrong_types(self):
        """Test schema validation with wrong data types."""
        invalid_data = pd.DataFrame({
            'counter_id': ['C1'],
            'counter_name': ['Test Counter'],
            'site_id': [1],
            'site_name': ['Test Site'],
            'bike_count': ['not_numeric'],  # Should be numeric
            'date': pd.date_range('2023-01-01', periods=1, freq='H'),
            'counter_installation_date': pd.date_range('2023-01-01', periods=1, freq='H'),
            'coordinates': ['48.8566,2.3522'],
            'counter_technical_id': ['T1'],
            'latitude': [48.8566],
            'longitude': [2.3522]
        })
        
        with pytest.raises(ValueError, match="'bike_count' column must be numeric"):
            validate_data_schema(invalid_data)
