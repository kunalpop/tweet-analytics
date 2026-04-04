import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def _read_csv_any(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    try:
        return pd.read_csv(p, engine="pyarrow")
    except Exception:
        return pd.read_csv(p)


def _detect_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    return None


def load_tweets_csv(path: Union[str, Path], text_column: Optional[str] = None, time_column: Optional[str] = None) -> pd.DataFrame:
    df = _read_csv_any(path)
    initial_rows = len(df)
    print(f"Parsed {initial_rows} rows from {path}")
    cols = df.columns.tolist()
    if text_column is None:
        text_column = _detect_column(cols, ["text", "full_text", "content", "body", "message"])
    if time_column is None:
        time_column = _detect_column(cols, ["created_at", "time", "timestamp", "date", "datetime"])
    if text_column is None or time_column is None:
        raise ValueError("Missing required text or time column")
    s_text = df[text_column].astype(str)
    s_time = df[time_column]
    if is_numeric_dtype(s_time):
        s = pd.to_numeric(s_time, errors="coerce")
        mx = s.max(skipna=True)
        if pd.isna(mx):
            t = pd.to_datetime(s, utc=True, errors="coerce")
        else:
            if mx > 10**12:
                t = pd.to_datetime((s // 1000).astype("Int64"), unit="s", utc=True, errors="coerce")
            elif mx > 10**10:
                t = pd.to_datetime((s / 1000), unit="s", utc=True, errors="coerce")
            else:
                t = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    elif is_datetime64_any_dtype(s_time):
        try:
            if getattr(s_time.dt, "tz", None) is not None:
                t = s_time.dt.tz_convert("UTC")
            else:
                t = s_time.dt.tz_localize("UTC")
        except Exception:
            t = pd.to_datetime(s_time, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(s_time, utc=True, errors="coerce")
    out = pd.DataFrame({"text": s_text, "created_at": t})
    
    rows_before_time_drop = len(out)
    out = out.dropna(subset=["created_at"])
    dropped_time = rows_before_time_drop - len(out)
    if dropped_time > 0:
        print(f"Dropped {dropped_time} rows due to missing or invalid timestamps")
        
    out["text"] = out["text"].str.strip()
    
    rows_before_text_drop = len(out)
    out = out[out["text"] != ""]
    dropped_text = rows_before_text_drop - len(out)
    if dropped_text > 0:
        print(f"Dropped {dropped_text} rows due to empty text")
        
    out = out.reset_index(drop=True)
    print(f"Successfully loaded {len(out)} clean rows")
    return out
