"""
Prediction models for bike counter forecasting.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from ..data.preprocess import create_date_encoder, create_full_pipeline
from ..data.loader import load_train_data, load_test_data


def create_ridge_model(alpha: float = 1.0) -> Ridge:
    """
    Create a Ridge regression model.
    
    Args:
        alpha: Regularization strength
        
    Returns:
        Ridge: Ridge regression model
    """
    return Ridge(alpha=alpha, random_state=42)


def create_random_forest_model(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Create a Random Forest regression model.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        random_state: Random state for reproducibility
        
    Returns:
        RandomForestRegressor: Random Forest model
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )


def create_bike_counter_pipeline(
    model_type: str = "ridge",
    alpha: float = 1.0,
    categorical_cols: list = ["counter_name", "site_name"]
) -> Pipeline:
    """
    Create a complete pipeline for bike counter prediction.
    
    Args:
        model_type: Type of model ("ridge" or "random_forest")
        alpha: Regularization strength for Ridge
        categorical_cols: Categorical columns to encode
        
    Returns:
        Pipeline: Complete prediction pipeline
    """
    # Date encoder
    date_encoder = create_date_encoder()
    
    # Get date columns after encoding
    sample_data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=1, freq="H"),
        "counter_name": ["test"],
        "site_name": ["test"]
    })
    
    encoded_sample = date_encoder.fit_transform(sample_data[["date"]])
    date_cols = encoded_sample.columns.tolist()
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ], remainder="passthrough")
    
    # Model
    if model_type == "ridge":
        model = create_ridge_model(alpha=alpha)
    elif model_type == "random_forest":
        model = create_random_forest_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Complete pipeline
    pipeline = Pipeline([
        ("date_encoder", date_encoder),
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5
) -> Tuple[Pipeline, dict]:
    """
    Train a model and return performance metrics.
    
    Args:
        pipeline: Model pipeline to train
        X_train: Training features
        y_train: Training targets
        cv_folds: Number of CV folds
        
    Returns:
        tuple: (trained_pipeline, metrics_dict)
    """
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Cross-validation
    cv = TimeSeriesSplit(n_splits=cv_folds)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=cv, 
        scoring="neg_root_mean_squared_error"
    )
    
    # Training predictions
    y_pred_train = pipeline.predict(X_train)
    
    # Metrics
    metrics = {
        "train_rmse": mean_squared_error(y_train, y_pred_train, squared=False),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "train_r2": r2_score(y_train, y_pred_train),
        "cv_rmse_mean": -cv_scores.mean(),
        "cv_rmse_std": cv_scores.std(),
        "cv_scores": -cv_scores
    }
    
    return pipeline, metrics


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate model performance on test data.
    
    Args:
        pipeline: Trained model pipeline
        X_test: Test features
        y_test: Test targets
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "test_rmse": mean_squared_error(y_test, y_pred, squared=False),
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_r2": r2_score(y_test, y_pred)
    }
    
    return metrics


def hyperparameter_tuning(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv_folds: int = 3
) -> Tuple[Pipeline, dict]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        pipeline: Base pipeline
        X_train: Training features
        y_train: Training targets
        param_grid: Parameter grid for tuning
        cv_folds: Number of CV folds
        
    Returns:
        tuple: (best_pipeline, best_params)
    """
    cv = TimeSeriesSplit(n_splits=cv_folds)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


def predict_bike_counts(
    pipeline: Pipeline,
    X: pd.DataFrame,
    return_log: bool = False
) -> np.ndarray:
    """
    Make predictions for bike counts.
    
    Args:
        pipeline: Trained model pipeline
        X: Features for prediction
        return_log: Whether to return log-transformed predictions
        
    Returns:
        np.ndarray: Predictions
    """
    predictions = pipeline.predict(X)
    
    if not return_log:
        # Transform back to original scale
        predictions = np.exp(predictions) - 1
    
    return predictions


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Get feature importance from the trained model.
    
    Args:
        pipeline: Trained model pipeline
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    model = pipeline.named_steps["model"]
    
    if hasattr(model, "feature_importances_"):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Ridge regression
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not support feature importance")
    
    # Get feature names from preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    return importance_df


def create_baseline_model() -> Pipeline:
    """
    Create a baseline model that predicts the mean.
    
    Returns:
        Pipeline: Baseline model pipeline
    """
    from sklearn.dummy import DummyRegressor
    
    baseline = Pipeline([
        ("model", DummyRegressor(strategy="mean"))
    ])
    
    return baseline
