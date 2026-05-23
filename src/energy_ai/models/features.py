import pandas as pd


def make_time_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)

    out["hour"] = out[ts_col].dt.hour.astype("int16")
    out["dow"] = out[ts_col].dt.dayofweek.astype("int16")
    out["month"] = out[ts_col].dt.month.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    return out


def add_lags(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts")
    for l in lags:
        out[f"{target_col}_lag_{l}h"] = out[target_col].shift(l)
    return out


def add_rolling(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts")
    for w in windows:
        out[f"{target_col}_rollmean_{w}h"] = out[target_col].rolling(w).mean()
    return out