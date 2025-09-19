from src.bike_counters.data.preprocess import normalize_columns


def test_normalize_columns(sample_counts_df):
    df = sample_counts_df.rename(columns={"Count": "Count"})
    out = normalize_columns(df)
    assert all(c == c.lower() for c in out.columns)
