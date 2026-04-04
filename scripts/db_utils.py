import sqlite3
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, List
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

def connect_sqlite(p: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(p))

def list_tables(conn: sqlite3.Connection) -> List[str]:
    try:
        df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
        return df["name"].tolist()
    except Exception:
        return []

def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        df = pd.read_sql_query(f"PRAGMA table_info('{table}')", conn)
        return df["name"].tolist()
    except Exception:
        return []

def coerce_datetime(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        mx = s.max(skipna=True)
        if pd.isna(mx):
            return pd.to_datetime(s, utc=True, errors="coerce")
        if mx > 10**12:
            return pd.to_datetime((s // 1000).astype("Int64"), unit="s", utc=True, errors="coerce")
        if mx > 10**10:
            return pd.to_datetime((s / 1000), unit="s", utc=True, errors="coerce")
        return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    if is_datetime64_any_dtype(series):
        try:
            if getattr(series.dt, "tz", None) is not None:
                return series.dt.tz_convert("UTC")
            return series.dt.tz_localize("UTC")
        except Exception:
            return pd.to_datetime(series, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")

@st.cache_data(show_spinner="Loading data from SQLite...")
def load_from_table(db_path: str, table: str, text_col: str, time_col: str, user_col: Optional[str]) -> pd.DataFrame:
    p = Path(db_path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    conn = connect_sqlite(p)
    try:
        cols = [text_col, time_col] + ([user_col] if user_col else [])
        col_sel = ", ".join([f'"{c}"' for c in cols])
        df = pd.read_sql_query(f'SELECT {col_sel} FROM "{table}"', conn)
    finally:
        conn.close()
    df.columns = ["text", "created_at"] + (["user"] if user_col else [])
    df["created_at"] = coerce_datetime(df["created_at"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["created_at"])
    df = df[df["text"] != ""]
    df = df.reset_index(drop=True)
    return df

def pick_default(cols: List[str], candidates: List[str]) -> Optional[str]:
    lc = [c.lower() for c in cols]
    for cand in candidates:
        if cand in lc:
            return cols[lc.index(cand)]
    return cols[0] if cols else None
