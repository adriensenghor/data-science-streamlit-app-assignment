import pandas as pd


def add_hour(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df["hour"] = pd.to_datetime(df[timestamp_col]).dt.hour
    return df
