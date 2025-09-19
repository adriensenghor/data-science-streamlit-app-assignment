# Bike Counter Prediction Dashboard

A comprehensive machine learning project for predicting bike counter traffic in Paris using Streamlit, scikit-learn, and modern data science tools.

## 🚴 Project Overview

This project provides analysis and prediction capabilities for bike counter data from Paris. It includes a complete machine learning pipeline with data preprocessing, feature engineering, model training, and interactive visualization through a Streamlit dashboard.

## 📊 Features

- **📈 Time Series Analysis**: Comprehensive analysis of bike counter patterns
- **🤖 Machine Learning**: Ridge regression and Random Forest models with proper pipelines
- **🗺️ Interactive Maps**: Folium-based maps showing counter locations
- **📊 Data Visualization**: Multiple chart types for data exploration
- **🔮 Predictive Modeling**: Real-time predictions with model evaluation
- **🧪 Comprehensive Testing**: Unit and integration tests for all modules
- **🐳 Docker Support**: Containerized deployment with Docker and docker-compose

## 🏗️ Project Structure

```
data-science-streamlit-app-assignment/
├── app/                          # Streamlit application
│   ├── streamlit_app.py         # Main dashboard
│   ├── pages/                    # Additional pages
│   ├── components/               # Reusable components
│   └── assets/                   # Static assets
├── src/bike_counters/            # Main package
│   ├── data/                     # Data handling
│   │   ├── loader.py            # Data loading utilities
│   │   ├── preprocess.py        # Preprocessing pipelines
│   │   └── validators.py        # Data validation
│   ├── features/                 # Feature engineering
│   │   └── engineering.py       # Feature creation utilities
│   ├── models/                   # ML models
│   │   └── predict.py           # Model training and prediction
│   └── viz/                      # Visualization
│       └── plotting.py          # Plotting utilities
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── data/sample/             # Test data
├── notebooks/                    # Jupyter notebooks
├── data/                         # Data directory (not tracked)
├── scripts/                      # Utility scripts
├── docker/                       # Docker configuration
└── .github/workflows/            # CI/CD pipeline
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data-science-streamlit-app-assignment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app/streamlit_app.py
   ```

### Using Docker

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**
   ```bash
   docker build -t bike-counter-app .
   docker run -p 8501:8501 bike-counter-app
   ```

## 📋 Usage

### Data Requirements

The application expects bike counter data in Parquet format:

- **Training data**: `data/train.parquet`
- **Test data**: `data/test.parquet` (optional)

Expected columns:
- `counter_id`, `counter_name`, `site_id`, `site_name`
- `bike_count`, `date`, `counter_installation_date`
- `coordinates`, `counter_technical_id`
- `latitude`, `longitude`, `log_bike_count`

### Dashboard Features

1. **Overview**: Project statistics and quick insights
2. **Data Analysis**: Interactive exploration with filters and visualizations
3. **Model Training**: Train Ridge or Random Forest models with hyperparameter tuning
4. **Predictions**: Make predictions on test data or custom inputs
5. **Model Evaluation**: Comprehensive performance analysis

### Command Line Interface

```bash
# Run tests
make unit          # Unit tests only
make all-tests     # All tests

# Code quality
make lint          # Lint with ruff
make format        # Format with black

# Development
make setup         # Install dependencies
make run           # Run Streamlit app
```

## 🧪 Testing

The project includes comprehensive tests:

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test module interactions and app startup
- **Data Validation**: Schema validation and error handling

Run tests:
```bash
pytest tests/ -v
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## 🔧 Development

### Code Quality

- **Linting**: Ruff for fast Python linting
- **Formatting**: Black for consistent code formatting
- **Pre-commit**: Automated quality checks

Setup pre-commit hooks:
```bash
pre-commit install
```

### Adding New Features

1. **Data Processing**: Add functions to `src/bike_counters/data/`
2. **Feature Engineering**: Extend `src/bike_counters/features/engineering.py`
3. **Models**: Add new models to `src/bike_counters/models/predict.py`
4. **Visualizations**: Create new plots in `src/bike_counters/viz/plotting.py`
5. **Tests**: Add corresponding tests in `tests/unit/`

## 📊 Model Performance

The project implements several models:

- **Ridge Regression**: Linear model with L2 regularization
- **Random Forest**: Ensemble method for non-linear patterns
- **Baseline**: Mean prediction for comparison

Key metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

## 🐳 Deployment

### Docker

The application is containerized for easy deployment:

```bash
# Build image
docker build -t bike-counter-app .

# Run container
docker run -p 8501:8501 bike-counter-app
```

### Environment Variables

- `DATA_DIR`: Path to data directory (default: `data`)
- `PORT`: Port for Streamlit (default: `8501`)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Paris bike counter data from city council
- RAMP (Rapid Analytics and Model Prototyping) framework
- Streamlit for the web interface
- scikit-learn for machine learning tools

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the documentation in the notebooks
- Review the test cases for usage examples
