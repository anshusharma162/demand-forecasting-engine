"""Feature engineering.

The model forecasts demand up to ``horizon`` days ahead using a *direct*
strategy: every lag / rolling feature is shifted by at least ``horizon`` days,
so all inputs are already known on the day the forecast is made. Reservations
and holidays are known in advance, so they are used for the target day itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["location_id", "area", "store_type"]


def build_panel(tables: dict[str, pd.DataFrame], horizon: int = 7) -> pd.DataFrame:
    """Merge all sources into one row per (location, date), including future dates."""
    sales = tables["sales"].copy()
    res = tables["reservations"].copy()
    hol = tables["holidays"].copy()
    loc = tables["locations"].copy()
    for df in (sales, res, hol):
        df["date"] = pd.to_datetime(df["date"])

    last_date = sales.date.max() + pd.Timedelta(days=horizon)
    dates = pd.date_range(sales.date.min(), last_date, freq="D")
    grid = pd.MultiIndex.from_product([loc.location_id, dates], names=["location_id", "date"])
    panel = pd.DataFrame(index=grid).reset_index()

    panel = panel.merge(sales, on=["location_id", "date"], how="left")
    panel = panel.merge(res, on=["location_id", "date"], how="left")
    panel["reserve_visitors"] = panel["reserve_visitors"].fillna(0)
    panel = panel.merge(loc, on="location_id", how="left")

    hol_dates = set(hol.date)
    panel["is_holiday"] = panel.date.isin(hol_dates).astype(int)
    panel["is_pre_holiday"] = (panel.date + pd.Timedelta(days=1)).isin(hol_dates).astype(int)
    panel["is_post_holiday"] = (panel.date - pd.Timedelta(days=1)).isin(hol_dates).astype(int)
    panel["is_history"] = panel.date <= sales.date.max()
    return panel.sort_values(["location_id", "date"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    df = panel.copy()

    # Calendar
    df["dow"] = df.date.dt.dayofweek
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["month"] = df.date.dt.month
    df["day_of_month"] = df.date.dt.day
    df["week_of_year"] = df.date.dt.isocalendar().week.astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df.date.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.date.dt.dayofyear / 365.25)

    # Reservations relative to outlet capacity
    df["reserve_ratio"] = df.reserve_visitors / df.capacity

    # Lags & rolling stats on log demand, shifted by >= horizon (no leakage)
    df["log_visitors"] = np.log1p(df.visitors)
    g = df.groupby("location_id")["log_visitors"]
    lags = sorted({horizon} | {l for l in (7, 14, 21, 28) if l >= horizon})
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(horizon)
    sg = shifted.groupby(df.location_id)
    for w in (7, 14, 28, 56):
        df[f"roll_mean_{w}"] = sg.transform(lambda s, w=w: s.rolling(w, min_periods=max(1, w // 2)).mean())
    df["roll_std_28"] = sg.transform(lambda s: s.rolling(28, min_periods=7).std())
    df["ewm_14"] = sg.transform(lambda s: s.ewm(span=14, ignore_na=True).mean())

    # Same weekday average over the last 4 available weeks
    same_dow = [c for c in df.columns if c.startswith("lag_") and int(c.split("_")[1]) % 7 == 0]
    df["same_dow_mean"] = df[same_dow].mean(axis=1)

    # Reservations momentum: bookings vs this outlet's recent booking average
    rg = df.groupby("location_id")["reserve_visitors"]
    df["reserve_roll_28"] = rg.transform(lambda s: s.shift(horizon).rolling(28, min_periods=7).mean())
    df["reserve_vs_avg"] = df.reserve_visitors / (df.reserve_roll_28 + 1)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"date", "visitors", "log_visitors", "is_history"}
    return [c for c in df.columns if c not in drop]
