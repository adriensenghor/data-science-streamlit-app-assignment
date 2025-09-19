"""
Tests for visualization utilities.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import folium

from src.bike_counters.viz.plotting import (
    line_counts,
    plot_time_series,
    plot_aggregated_counts,
    plot_distribution,
    create_counter_map,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_hourly_patterns,
    plot_weekly_patterns
)


class TestVisualization:
    """Test cases for visualization functions."""
    
    def test_line_counts(self):
        """Test line chart creation with Altair."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='H'),
            'bike_count': [10, 12, 8, 15, 11]
        })
        
        chart = line_counts(df)
        
        assert isinstance(chart, alt.Chart)
        assert chart.mark == 'line'
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        df = pd.DataFrame({
            'counter_name': ['C1'] * 5,
            'date': pd.date_range('2023-01-01', periods=5, freq='H'),
            'bike_count': [10, 12, 8, 15, 11]
        })
        
        fig = plot_time_series(df, 'C1')
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_time_series_no_data(self):
        """Test time series plotting with no data."""
        df = pd.DataFrame({
            'counter_name': ['C2'] * 5,
            'date': pd.date_range('2023-01-01', periods=5, freq='H'),
            'bike_count': [10, 12, 8, 15, 11]
        })
        
        with pytest.raises(ValueError, match="No data found for counter"):
            plot_time_series(df, 'C1')
    
    def test_plot_aggregated_counts(self):
        """Test aggregated counts plotting."""
        df = pd.DataFrame({
            'counter_name': ['C1'] * 10,
            'date': pd.date_range('2023-01-01', periods=10, freq='H'),
            'bike_count': range(10)
        })
        
        fig = plot_aggregated_counts(df, 'C1', freq='1D')
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_distribution(self):
        """Test distribution plotting."""
        df = pd.DataFrame({
            'bike_count': np.random.poisson(10, 100)
        })
        
        fig = plot_distribution(df, 'bike_count')
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_distribution_log_scale(self):
        """Test distribution plotting with log scale."""
        df = pd.DataFrame({
            'bike_count': np.random.poisson(10, 100)
        })
        
        fig = plot_distribution(df, 'bike_count', log_scale=True)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_create_counter_map(self):
        """Test counter map creation."""
        df = pd.DataFrame({
            'counter_name': ['C1', 'C2'],
            'latitude': [48.8566, 48.8600],
            'longitude': [2.3522, 2.3600]
        })
        
        map_obj = create_counter_map(df)
        
        assert isinstance(map_obj, folium.Map)
    
    def test_plot_predictions_vs_actual(self):
        """Test predictions vs actual plotting."""
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.normal(0, 0.1, 50)
        
        fig = plot_predictions_vs_actual(y_true, y_pred)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_residuals(self):
        """Test residuals plotting."""
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.normal(0, 0.1, 50)
        
        fig = plot_residuals(y_true, y_pred)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'importance': np.random.rand(10)
        }).sort_values('importance', ascending=False)
        
        fig = plot_feature_importance(importance_df, top_n=5)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_hourly_patterns(self):
        """Test hourly patterns plotting."""
        df = pd.DataFrame({
            'counter_name': ['C1'] * 48,
            'date': pd.date_range('2023-01-01', periods=48, freq='H'),
            'bike_count': np.random.poisson(10, 48)
        })
        
        fig = plot_hourly_patterns(df, 'C1')
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_hourly_patterns_all_counters(self):
        """Test hourly patterns plotting for all counters."""
        df = pd.DataFrame({
            'counter_name': ['C1', 'C2'] * 24,
            'date': pd.date_range('2023-01-01', periods=48, freq='H'),
            'bike_count': np.random.poisson(10, 48)
        })
        
        fig = plot_hourly_patterns(df)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_weekly_patterns(self):
        """Test weekly patterns plotting."""
        df = pd.DataFrame({
            'counter_name': ['C1'] * 168,  # 7 days * 24 hours
            'date': pd.date_range('2023-01-01', periods=168, freq='H'),
            'bike_count': np.random.poisson(10, 168)
        })
        
        fig = plot_weekly_patterns(df, 'C1')
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
