"""
Bike Counter Prediction Dashboard
A streamlined Streamlit app for bike counter analysis and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Bike Counter Prediction Dashboard",
    page_icon="🚴",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚴 Bike Counter Prediction Dashboard</h1>', unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'counter_name': np.random.choice(['Counter A', 'Counter B', 'Counter C'], 1000),
        'date': dates,
        'bike_count': np.random.poisson(15, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 24) * 5,
        'latitude': np.random.uniform(48.8, 48.9, 1000),
        'longitude': np.random.uniform(2.3, 2.4, 1000)
    })
    
    # Add some temporal patterns
    data['hour'] = data['date'].dt.hour
    data['weekday'] = data['date'].dt.weekday
    data['bike_count'] = data['bike_count'] + (data['hour'] - 12) * 2
    data['bike_count'] = np.maximum(data['bike_count'], 0)
    
    return data

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Analysis", "🤖 Prediction"])

with tab1:
    st.header("Data Overview")
    
    # Load and display data
    data = load_sample_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Unique Counters", f"{data['counter_name'].nunique():,}")
    
    with col3:
        st.metric("Date Range", f"{(data['date'].max() - data['date'].min()).days} days")
    
    with col4:
        st.metric("Avg Bike Count", f"{data['bike_count'].mean():.1f}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Counter selection
    selected_counter = st.selectbox("Select Counter", ["All"] + list(data['counter_name'].unique()))
    
    if selected_counter != "All":
        filtered_data = data[data['counter_name'] == selected_counter]
    else:
        filtered_data = data
    
    # Time series plot
    st.subheader("Bike Counts Over Time")
    fig = px.line(filtered_data, x='date', y='bike_count', 
                  title=f"Bike Counts - {selected_counter}")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Analysis")
    
    data = load_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Patterns")
        hourly_avg = data.groupby('hour')['bike_count'].mean()
        fig_hourly = px.bar(x=hourly_avg.index, y=hourly_avg.values,
                           title="Average Bike Count by Hour")
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.subheader("Weekly Patterns")
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = data.groupby('weekday')['bike_count'].mean()
        fig_weekly = px.bar(x=[weekday_names[i] for i in weekly_avg.index], 
                           y=weekly_avg.values,
                           title="Average Bike Count by Day of Week")
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Distribution
    st.subheader("Bike Count Distribution")
    fig_dist = px.histogram(data, x='bike_count', nbins=30, 
                           title="Distribution of Bike Counts")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Map
    st.subheader("Counter Locations")
    map_data = data[['counter_name', 'latitude', 'longitude']].drop_duplicates('counter_name')
    fig_map = px.scatter_mapbox(map_data, lat="latitude", lon="longitude",
                               hover_name="counter_name", zoom=11, height=500)
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.header("Simple Prediction Model")
    
    data = load_sample_data()
    
    st.info("This is a demonstration with sample data. In a real implementation, you would train ML models on actual bike counter data.")
    
    # Simple prediction interface
    col1, col2 = st.columns(2)
    
    with col1:
        counter_name = st.selectbox("Counter", data['counter_name'].unique())
        hour = st.slider("Hour of Day", 0, 23, 12)
        weekday = st.selectbox("Day of Week", weekday_names)
    
    with col2:
        temperature = st.slider("Temperature (°C)", -10, 40, 15)
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Snowy"])
    
    if st.button("Predict Bike Count"):
        # Simple heuristic prediction
        base_count = 10
        
        # Hour effect
        hour_effect = np.sin(hour * 2 * np.pi / 24) * 5
        
        # Weather effect
        weather_effects = {"Sunny": 2, "Cloudy": 0, "Rainy": -3, "Snowy": -5}
        weather_effect = weather_effects.get(weather, 0)
        
        # Temperature effect
        temp_effect = (temperature - 15) * 0.2
        
        # Counter effect
        counter_effects = {"Counter A": 3, "Counter B": 0, "Counter C": -2}
        counter_effect = counter_effects.get(counter_name, 0)
        
        prediction = base_count + hour_effect + weather_effect + temp_effect + counter_effect
        prediction = max(0, int(prediction))
        
        st.success(f"Predicted bike count: **{prediction}**")
        
        # Show prediction breakdown
        st.subheader("Prediction Breakdown")
        breakdown_data = {
            "Component": ["Base Count", "Hour Effect", "Weather Effect", "Temperature Effect", "Counter Effect"],
            "Value": [base_count, hour_effect, weather_effect, temp_effect, counter_effect]
        }
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Bike Counter Prediction Dashboard | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)