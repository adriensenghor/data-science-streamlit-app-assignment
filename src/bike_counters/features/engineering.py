"""
Feature engineering utilities for bike counter prediction.
"""

import pandas as pd
import numpy as np
from typing import Union, List


def encode_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Extract temporal features from date column.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        
    Returns:
        DataFrame: With temporal features extracted
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract temporal features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["hour"] = df[date_col].dt.hour
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    
    # Cyclical encoding for periodic features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def add_hour(df: pd.DataFrame, timestamp_col: str = "date") -> pd.DataFrame:
    """
    Add hour feature from timestamp column.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of the timestamp column
        
    Returns:
        DataFrame: With hour feature added
    """
    df = df.copy()
    df["hour"] = pd.to_datetime(df[timestamp_col]).dt.hour
    return df


def add_cyclical_features(df: pd.DataFrame, 
                         hour_col: str = "hour",
                         weekday_col: str = "weekday",
                         month_col: str = "month") -> pd.DataFrame:
    """
    Add cyclical encoding for periodic features.
    
    Args:
        df: DataFrame with periodic features
        hour_col: Name of hour column
        weekday_col: Name of weekday column  
        month_col: Name of month column
        
    Returns:
        DataFrame: With cyclical features added
    """
    df = df.copy()
    
    if hour_col in df.columns:
        df[f"{hour_col}_sin"] = np.sin(2 * np.pi * df[hour_col] / 24)
        df[f"{hour_col}_cos"] = np.cos(2 * np.pi * df[hour_col] / 24)
    
    if weekday_col in df.columns:
        df[f"{weekday_col}_sin"] = np.sin(2 * np.pi * df[weekday_col] / 7)
        df[f"{weekday_col}_cos"] = np.cos(2 * np.pi * df[weekday_col] / 7)
    
    if month_col in df.columns:
        df[f"{month_col}_sin"] = np.sin(2 * np.pi * df[month_col] / 12)
        df[f"{month_col}_cos"] = np.cos(2 * np.pi * df[month_col] / 12)
    
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather-related features (placeholder for external data integration).
    
    Args:
        df: DataFrame to add weather features to
        
    Returns:
        DataFrame: With weather features added
    """
    df = df.copy()
    
    # Placeholder weather features - would be replaced with actual weather data
    df["temperature"] = 15.0  # Default temperature
    df["precipitation"] = 0.0  # Default precipitation
    df["wind_speed"] = 5.0  # Default wind speed
    
    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add location-based features.
    
    Args:
        df: DataFrame with latitude and longitude columns
        
    Returns:
        DataFrame: With location features added
    """
    df = df.copy()
    
    if "latitude" in df.columns and "longitude" in df.columns:
        # Distance from city center (approximate Paris center)
        paris_center_lat, paris_center_lon = 48.8566, 2.3522
        
        df["distance_from_center"] = np.sqrt(
            (df["latitude"] - paris_center_lat) ** 2 + 
            (df["longitude"] - paris_center_lon) ** 2
        )
        
        # Zone classification based on distance
        df["zone"] = pd.cut(
            df["distance_from_center"], 
            bins=[0, 0.01, 0.02, 0.05, float('inf')],
            labels=["center", "inner", "outer", "suburbs"]
        )
    
    return df


def create_lag_features(df: pd.DataFrame, 
                        target_col: str = "bike_count",
                        lags: List[int] = [1, 2, 3, 24, 48]) -> pd.DataFrame:
    """
    Create lag features for time series prediction.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        lags: List of lag periods to create
        
    Returns:
        DataFrame: With lag features added
    """
    df = df.copy()
    
    # Sort by counter and date for proper lag calculation
    df = df.sort_values(["counter_name", "date"])
    
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby("counter_name")[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                          target_col: str = "bike_count",
                          windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        windows: List of window sizes for rolling features
        
    Returns:
        DataFrame: With rolling features added
    """
    df = df.copy()
    
    # Sort by counter and date
    df = df.sort_values(["counter_name", "date"])
    
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df.groupby("counter_name")[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        df[f"{target_col}_rolling_std_{window}"] = (
            df.groupby("counter_name")[target_col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
    
    return df
