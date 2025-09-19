"""
Tests for data validation utilities.
"""

import pytest
import pandas as pd
import numpy as np

from src.bike_counters.data.validators import expect_columns


class TestDataValidators:
    """Test cases for data validation functions."""
    
    def test_expect_columns_success(self):
        """Test successful column validation."""
        df = pd.DataFrame({
            'station_id': [1, 2],
            'timestamp': ['2023-01-01', '2023-01-02'],
            'count': [10, 12]
        })
        
        # Should not raise any exception
        expect_columns(df, ['station_id', 'timestamp', 'count'])
    
    def test_expect_columns_missing(self):
        """Test column validation with missing columns."""
        df = pd.DataFrame({
            'station_id': [1, 2],
            'count': [10, 12]
        })
        
        with pytest.raises(ValueError, match="Missing columns"):
            expect_columns(df, ['station_id', 'timestamp', 'count'])
    
    def test_expect_columns_empty_required(self):
        """Test column validation with empty required list."""
        df = pd.DataFrame({
            'station_id': [1, 2],
            'count': [10, 12]
        })
        
        # Should not raise any exception
        expect_columns(df, [])
    
    def test_expect_columns_subset(self):
        """Test column validation with subset of columns."""
        df = pd.DataFrame({
            'station_id': [1, 2],
            'timestamp': ['2023-01-01', '2023-01-02'],
            'count': [10, 12],
            'extra_col': ['A', 'B']
        })
        
        # Should not raise any exception
        expect_columns(df, ['station_id', 'count'])
