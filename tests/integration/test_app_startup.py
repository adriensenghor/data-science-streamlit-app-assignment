"""
Integration tests for app startup and basic functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestAppStartup:
    """Test cases for app startup and integration."""
    
    def test_import_streamlit_app(self):
        """Test that Streamlit app can be imported."""
        try:
            import app.streamlit_app
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import streamlit app: {e}")
    
    def test_import_bike_counters_package(self):
        """Test that bike_counters package can be imported."""
        try:
            from bike_counters.data.loader import load_train_data
            from bike_counters.data.preprocess import clean_data
            from bike_counters.features.engineering import encode_dates
            from bike_counters.models.predict import create_ridge_model
            from bike_counters.viz.plotting import plot_time_series
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import bike_counters package: {e}")
    
    def test_utils_module_import(self):
        """Test that utils module can be imported."""
        try:
            import utils
            assert hasattr(utils, 'get_train_data')
            assert hasattr(utils, 'encode_dates')
            assert hasattr(utils, 'train_test_split_temporal')
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import utils module: {e}")
    
    def test_package_structure(self):
        """Test that package structure is correct."""
        src_path = Path(__file__).parent.parent.parent / "src" / "bike_counters"
        
        # Check main directories exist
        assert (src_path / "data").exists()
        assert (src_path / "features").exists()
        assert (src_path / "models").exists()
        assert (src_path / "viz").exists()
        
        # Check main files exist
        assert (src_path / "__init__.py").exists()
        assert (src_path / "config.py").exists()
        assert (src_path / "data" / "loader.py").exists()
        assert (src_path / "data" / "preprocess.py").exists()
        assert (src_path / "data" / "validators.py").exists()
        assert (src_path / "features" / "engineering.py").exists()
        assert (src_path / "models" / "predict.py").exists()
        assert (src_path / "viz" / "plotting.py").exists()
