from src.bike_counters.data.loader import load_csv


def test_load_csv(sample_counts_df, tmp_path):
    p = tmp_path / "sample.csv"
    sample_counts_df.to_csv(p, index=False)
    df = load_csv(p)
    assert not df.empty
