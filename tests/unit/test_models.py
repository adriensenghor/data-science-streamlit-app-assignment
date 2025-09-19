"""
Tests for model prediction utilities.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from src.bike_counters.models.predict import (
    create_ridge_model,
    create_random_forest_model,
    create_bike_counter_pipeline,
    train_model,
    evaluate_model,
    predict_bike_counts,
    get_feature_importance,
    create_baseline_model
)


class TestModelPrediction:
    """Test cases for model prediction functions."""
    
    def test_create_ridge_model(self):
        """Test Ridge model creation."""
        model = create_ridge_model(alpha=2.0)
        
        assert isinstance(model, Ridge)
        assert model.alpha == 2.0
        assert model.random_state == 42
    
    def test_create_random_forest_model(self):
        """Test Random Forest model creation."""
        model = create_random_forest_model(n_estimators=50, max_depth=5)
        
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 42
    
    def test_create_bike_counter_pipeline_ridge(self):
        """Test bike counter pipeline creation with Ridge."""
        pipeline = create_bike_counter_pipeline(model_type="ridge", alpha=1.5)
        
        assert isinstance(pipeline, Pipeline)
        assert 'date_encoder' in pipeline.named_steps
        assert 'preprocessor' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
        assert isinstance(pipeline.named_steps['model'], Ridge)
    
    def test_create_bike_counter_pipeline_random_forest(self):
        """Test bike counter pipeline creation with Random Forest."""
        pipeline = create_bike_counter_pipeline(model_type="random_forest")
        
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps['model'], RandomForestRegressor)
    
    def test_create_bike_counter_pipeline_invalid_model(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_bike_counter_pipeline(model_type="invalid_model")
    
    def test_train_model(self):
        """Test model training functionality."""
        # Create test data
        X_train = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='H'),
            'counter_name': ['C1'] * 100,
            'site_name': ['S1'] * 100
        })
        y_train = pd.Series(np.random.randn(100))
        
        # Create and train pipeline
        pipeline = create_bike_counter_pipeline(model_type="ridge")
        trained_pipeline, metrics = train_model(pipeline, X_train, y_train, cv_folds=3)
        
        assert isinstance(trained_pipeline, Pipeline)
        assert 'train_rmse' in metrics
        assert 'train_mae' in metrics
        assert 'train_r2' in metrics
        assert 'cv_rmse_mean' in metrics
        assert 'cv_rmse_std' in metrics
        
        # Check that metrics are reasonable
        assert metrics['train_rmse'] >= 0
        assert metrics['train_r2'] <= 1.0
    
    def test_evaluate_model(self):
        """Test model evaluation functionality."""
        # Create test data
        X_test = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20, freq='H'),
            'counter_name': ['C1'] * 20,
            'site_name': ['S1'] * 20
        })
        y_test = pd.Series(np.random.randn(20))
        
        # Create and train pipeline
        pipeline = create_bike_counter_pipeline(model_type="ridge")
        pipeline.fit(X_test, y_test)
        
        # Evaluate model
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        assert 'test_rmse' in metrics
        assert 'test_mae' in metrics
        assert 'test_r2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['test_rmse'] >= 0
        assert metrics['test_r2'] <= 1.0
    
    def test_predict_bike_counts_log_scale(self):
        """Test bike count prediction in log scale."""
        # Create test data
        X = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='H'),
            'counter_name': ['C1'] * 5,
            'site_name': ['S1'] * 5
        })
        
        # Create and train pipeline
        pipeline = create_bike_counter_pipeline(model_type="ridge")
        y_train = pd.Series(np.random.randn(5))
        pipeline.fit(X, y_train)
        
        # Predict in log scale
        predictions = predict_bike_counts(pipeline, X, return_log=True)
        
        assert len(predictions) == 5
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_bike_counts_original_scale(self):
        """Test bike count prediction in original scale."""
        # Create test data
        X = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='H'),
            'counter_name': ['C1'] * 5,
            'site_name': ['S1'] * 5
        })
        
        # Create and train pipeline
        pipeline = create_bike_counter_pipeline(model_type="ridge")
        y_train = pd.Series(np.random.randn(5))
        pipeline.fit(X, y_train)
        
        # Predict in original scale
        predictions = predict_bike_counts(pipeline, X, return_log=False)
        
        assert len(predictions) == 5
        assert isinstance(predictions, np.ndarray)
        assert np.all(predictions >= 0)  # Bike counts should be non-negative
    
    def test_get_feature_importance_ridge(self):
        """Test feature importance extraction from Ridge model."""
        # Create test data
        X_train = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='H'),
            'counter_name': ['C1'] * 10,
            'site_name': ['S1'] * 10
        })
        y_train = pd.Series(np.random.randn(10))
        
        # Create and train pipeline
        pipeline = create_bike_counter_pipeline(model_type="ridge")
        pipeline.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = get_feature_importance(pipeline)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) > 0
    
    def test_get_feature_importance_random_forest(self):
        """Test feature importance extraction from Random Forest model."""
        # Create test data
        X_train = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='H'),
            'counter_name': ['C1'] * 10,
            'site_name': ['S1'] * 10
        })
        y_train = pd.Series(np.random.randn(10))
        
        # Create and train pipeline
        pipeline = create_bike_counter_pipeline(model_type="random_forest")
        pipeline.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = get_feature_importance(pipeline)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) > 0
    
    def test_create_baseline_model(self):
        """Test baseline model creation."""
        baseline = create_baseline_model()
        
        assert isinstance(baseline, Pipeline)
        assert 'model' in baseline.named_steps
