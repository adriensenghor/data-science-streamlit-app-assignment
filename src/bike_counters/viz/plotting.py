"""
Visualization utilities for bike counter data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import folium
from typing import Optional, List, Tuple, Union


def line_counts(df: pd.DataFrame, 
                x: str = "date", 
                y: str = "bike_count",
                title: str = "Bike Counts Over Time") -> alt.Chart:
    """
    Create a line chart of bike counts over time using Altair.
    
    Args:
        df: DataFrame with time series data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        
    Returns:
        alt.Chart: Altair line chart
    """
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X(x, title="Date"),
        y=alt.Y(y, title="Bike Count"),
        tooltip=[x, y]
    ).properties(
        title=title,
        width=600,
        height=300
    )
    
    return chart


def plot_time_series(df: pd.DataFrame,
                    counter_name: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot time series for a specific counter.
    
    Args:
        df: DataFrame with bike counter data
        counter_name: Name of the counter to plot
        start_date: Start date for filtering (optional)
        end_date: End date for filtering (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Filter data
    mask = df["counter_name"] == counter_name
    if start_date:
        mask &= df["date"] >= start_date
    if end_date:
        mask &= df["date"] <= end_date
    
    data = df[mask].copy()
    
    if data.empty:
        raise ValueError(f"No data found for counter: {counter_name}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(data["date"], data["bike_count"], linewidth=1)
    ax.set_title(f"Bike Counts - {counter_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bike Count")
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_aggregated_counts(df: pd.DataFrame,
                         counter_name: str,
                         freq: str = "1D",
                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot aggregated bike counts by frequency.
    
    Args:
        df: DataFrame with bike counter data
        counter_name: Name of the counter to plot
        freq: Aggregation frequency (e.g., '1D', '1W', '1H')
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Filter and aggregate data
    mask = df["counter_name"] == counter_name
    data = df[mask].copy()
    
    if data.empty:
        raise ValueError(f"No data found for counter: {counter_name}")
    
    # Aggregate by frequency
    aggregated = data.groupby(
        pd.Grouper(freq=freq, key="date")
    )["bike_count"].sum().reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(aggregated["date"], aggregated["bike_count"], linewidth=2)
    ax.set_title(f"Aggregated Bike Counts - {counter_name} ({freq})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Bike Count")
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_distribution(df: pd.DataFrame,
                     column: str = "bike_count",
                     bins: int = 50,
                     log_scale: bool = False,
                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot distribution of bike counts.
    
    Args:
        df: DataFrame with bike counter data
        column: Column to plot distribution for
        bins: Number of bins for histogram
        log_scale: Whether to use log scale
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if log_scale:
        data = np.log1p(df[column])
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f"Log({column})")
        ax.set_title(f"Distribution of Log({column})")
    else:
        ax.hist(df[column], bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel(column)
        ax.set_title(f"Distribution of {column}")
    
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_counter_map(df: pd.DataFrame,
                      zoom_start: int = 13,
                      center_lat: float = 48.8566,
                      center_lon: float = 2.3522) -> folium.Map:
    """
    Create a map showing bike counter locations.
    
    Args:
        df: DataFrame with bike counter data
        zoom_start: Initial zoom level
        center_lat: Center latitude
        center_lon: Center longitude
        
    Returns:
        folium.Map: Interactive map
    """
    # Get unique counter locations
    locations = df[["counter_name", "latitude", "longitude"]].drop_duplicates("counter_name")
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start
    )
    
    # Add markers for each counter
    for _, row in locations.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=row["counter_name"],
            tooltip=row["counter_name"]
        ).add_to(m)
    
    return m


def plot_predictions_vs_actual(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              title: str = "Predictions vs Actual",
                              figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   title: str = "Residuals Plot",
                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot residuals (errors) vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel("Predicted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Residuals")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot feature importance from model.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Get top features
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features["importance"])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_hourly_patterns(df: pd.DataFrame,
                        counter_name: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot hourly patterns of bike counts.
    
    Args:
        df: DataFrame with bike counter data
        counter_name: Specific counter to analyze (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Filter data if counter specified
    data = df.copy()
    if counter_name:
        data = data[data["counter_name"] == counter_name]
    
    # Extract hour from date
    data["hour"] = pd.to_datetime(data["date"]).dt.hour
    
    # Group by hour and calculate mean
    hourly_avg = data.groupby("hour")["bike_count"].mean()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Bike Count")
    ax.set_title(f"Hourly Patterns{' - ' + counter_name if counter_name else ''}")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24))
    
    plt.tight_layout()
    return fig


def plot_weekly_patterns(df: pd.DataFrame,
                        counter_name: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot weekly patterns of bike counts.
    
    Args:
        df: DataFrame with bike counter data
        counter_name: Specific counter to analyze (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Filter data if counter specified
    data = df.copy()
    if counter_name:
        data = data[data["counter_name"] == counter_name]
    
    # Extract weekday from date
    data["weekday"] = pd.to_datetime(data["date"]).dt.weekday
    
    # Group by weekday and calculate mean
    weekly_avg = data.groupby("weekday")["bike_count"].mean()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    ax.plot(range(7), weekly_avg.values, marker='o', linewidth=2)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Bike Count")
    ax.set_title(f"Weekly Patterns{' - ' + counter_name if counter_name else ''}")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(7))
    ax.set_xticklabels(weekday_names, rotation=45)
    
    plt.tight_layout()
    return fig
