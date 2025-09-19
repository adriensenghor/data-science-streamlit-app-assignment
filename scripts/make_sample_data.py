import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("tests/data/sample/sample_bike_counts.csv")
    df.to_csv("data/interim/sample.csv", index=False)
