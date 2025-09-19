"""
Utility functions for bike counter prediction project.
Based on the RAMP starting kit notebook.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def get_train_data():
    """
    Load training data from parquet file.
    
    Returns:
        tuple: (X, y) where X contains features and y contains log_bike_count target
    """
    data_path = Path("data") / "train.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    data = pd.read_parquet(data_path)
    
    # Separate features and target
    X = data.drop(columns=["log_bike_count"])
    y = data["log_bike_count"]
    
    return X, y


def get_test_data():
    """
    Load test data from parquet file.
    
    Returns:
        pd.DataFrame: Test features
    """
    data_path = Path("data") / "test.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found at {data_path}")
    
    return pd.read_parquet(data_path)


def train_test_split_temporal(X, y, delta_threshold="30 days"):
    """
    Split data temporally, keeping the last period for validation.
    
    Args:
        X: Features dataframe
        y: Target series
        delta_threshold: Time period to reserve for validation
        
    Returns:
        tuple: (X_train, y_train, X_valid, y_valid)
    """
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid


def encode_dates(X):
    """
    Extract temporal features from date column.
    
    Args:
        X: DataFrame with 'date' column
        
    Returns:
        DataFrame: With temporal features extracted
    """
    X = X.copy()
    # Encode the date information from the date column
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Drop the original date column
    return X.drop(columns=["date"])


def log_transform_predictions(y_pred_log):
    """
    Transform log predictions back to original bike count scale.
    
    Args:
        y_pred_log: Predictions in log scale
        
    Returns:
        array: Predictions in original bike count scale
    """
    return np.exp(y_pred_log) - 1
