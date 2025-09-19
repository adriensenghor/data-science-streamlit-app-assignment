# Bike Counter Prediction Dashboard

A clean and streamlined Streamlit application for bike counter data analysis and prediction.

## 🚴 Project Overview

This project provides a simple yet comprehensive dashboard for analyzing bike counter data with interactive visualizations and basic prediction capabilities.

## 📊 Features

- **📈 Data Visualization**: Interactive charts with Plotly
- **📊 Data Analysis**: Hourly and weekly pattern analysis
- **🔮 Simple Predictions**: Heuristic-based bike count prediction
- **🗺️ Location Maps**: Counter location visualization
- **🧪 Testing**: Unit tests for core functionality
- **🐳 Docker Support**: Easy containerized deployment

## 🏗️ Project Structure

```
data-science-streamlit-app-assignment/
├── app/
│   └── streamlit_app.py         # Main Streamlit application
├── src/bike_counters/            # Core package (optional for advanced features)
├── tests/                        # Test suite
├── notebooks/
│   └── bike_counters_starting_kit.ipynb  # Original notebook
├── data/                         # Data directory (not tracked)
├── docker/                       # Docker configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data-science-streamlit-app-assignment
   ```

2. **Install dependencies**
   ```bash
   make setup
   # or
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   make run
   # or
   streamlit run app/streamlit_app.py
   ```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or manually
make docker-build
make docker-run
```

## 📋 Usage

The dashboard includes three main tabs:

1. **📊 Data Overview**: View data statistics and time series
2. **📈 Analysis**: Explore hourly/weekly patterns and distributions
3. **🤖 Prediction**: Make simple predictions based on time and weather

### Sample Data

The app includes sample data for demonstration. In a real implementation, you would:
- Load actual bike counter data
- Implement proper ML models
- Connect to real-time data sources

## 🧪 Testing

Run the test suite:
```bash
make test
# or
pytest tests/ -v
```

## 🔧 Development

### Code Quality

```bash
make lint      # Lint with ruff
make format    # Format with black
make clean     # Clean up cache files
```

### Adding Features

The app is designed to be easily extensible:
- Add new visualizations in the Analysis tab
- Implement ML models in the Prediction tab
- Extend the data loading functions

## 🐳 Deployment

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run
```

The app will be available at `http://localhost:8501`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the RAMP bike counter starting kit
- Built with Streamlit and Plotly
- Sample data generated for demonstration purposes
