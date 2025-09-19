import pandas as pd
from src.bike_counters.data.validators import expect_columns


def test_expect_columns_ok(sample_counts_df):
    expect_columns(sample_counts_df, ["station_id", "timestamp", "count"])


def test_expect_columns_missing():
    df = pd.DataFrame({"a": [1]})
    try:
        expect_columns(df, ["a", "b"])
    except ValueError as e:
        assert "Missing columns" in str(e)
    else:
        assert False, "Expected ValueError"
