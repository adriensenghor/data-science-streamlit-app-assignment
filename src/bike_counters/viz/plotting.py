import pandas as pd
import altair as alt


def line_counts(df: pd.DataFrame, x: str = "timestamp", y: str = "count") -> alt.Chart:
    return alt.Chart(df).mark_line().encode(x=x, y=y)
