import pandas as pd


def filter_event_window(df: pd.DataFrame, start: str = "2012-07-01", end: str = "2012-07-07") -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    m = (df["created_at"] >= s) & (df["created_at"] <= e)
    return df.loc[m].reset_index(drop=True)
