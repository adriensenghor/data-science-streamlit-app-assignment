import pandas as pd
import pytest


@pytest.fixture
def sample_counts_df():
    return pd.read_csv("tests/data/sample/sample_bike_counts.csv")
