"""
data_prep.py — M5 data loading, melting, and feature engineering utilities.

This module is intentionally STABLE — the agent should not modify it.
Feature engineering lives in train.py so the agent can iterate freely.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    SALES_FILE, CALENDAR_FILE, PRICES_FILE,
    N_TRAIN_DAYS, VAL_START_DAY, VAL_END_DAY, HORIZON,
)


# ─────────────────────────────────────────────────────────────────────────────
# Raw data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_sales() -> pd.DataFrame:
    """Load sales_train_validation.csv and return wide format."""
    print("[data_prep] Loading sales …")
    df = pd.read_csv(SALES_FILE)
    return df


def load_calendar() -> pd.DataFrame:
    """Load calendar.csv with useful dtypes."""
    print("[data_prep] Loading calendar …")
    cal = pd.read_csv(CALENDAR_FILE)
    cal["date"] = pd.to_datetime(cal["date"])
    # Encode events as booleans
    cal["event_flag"]  = cal["event_name_1"].notna().astype(np.int8)
    cal["event2_flag"] = cal["event_name_2"].notna().astype(np.int8)
    # SNAP flags
    for state in ["CA", "TX", "WI"]:
        cal[f"snap_{state}"] = cal[f"snap_{state}"].astype(np.int8)
    return cal


def load_prices() -> pd.DataFrame:
    """Load sell_prices.csv."""
    print("[data_prep] Loading prices …")
    prices = pd.read_csv(PRICES_FILE)
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Melt to long format
# ─────────────────────────────────────────────────────────────────────────────

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sales df to long format with day integer column."""
    print("[data_prep] Melting sales to long format …")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    df = sales[ID_COLS + day_cols].melt(
        id_vars=ID_COLS,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    df["d"] = df["d"].str[2:].astype(np.int16)
    df["sales"] = df["sales"].astype(np.float32)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merge calendar and prices
# ─────────────────────────────────────────────────────────────────────────────

def merge_calendar(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Merge calendar features onto long-format sales df."""
    cal_cols = [
        "d", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_flag", "event2_flag",
        "snap_CA", "snap_TX", "snap_WI",
        "date",
    ]
    cal_sub = calendar[cal_cols].copy()
    cal_sub["d"] = cal_sub["d"].str[2:].astype(np.int16)
    return df.merge(cal_sub, on="d", how="left")


def merge_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Merge sell prices onto long-format df."""
    return df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Categorical encodings
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode string ID columns to integers for tree models."""
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype(np.int16)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Lag safety mask — prevents leakage into validation window
# ─────────────────────────────────────────────────────────────────────────────

def apply_lag_safety_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zero out lag features that would cause leakage into the validation window.
    For validation day d, lag_n is safe only if d - n <= VAL_START_DAY - 1.
    """
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    for col in lag_cols:
        n = int(col.split("_")[1])
        unsafe_days = [d for d in range(VAL_START_DAY, VAL_END_DAY + 1)
                      if d - n >= VAL_START_DAY]
        if unsafe_days:
            mask = df["d"].isin(unsafe_days)
            df.loc[mask, col] = np.nan
    return df



# ─────────────────────────────────────────────────────────────────────────────
# WRMSSE weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_wrmsse_weights(sales: pd.DataFrame, prices: pd.DataFrame,
                           calendar: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    print("[data_prep] Computing WRMSSE weights …")
    day_cols = [f"d_{i}" for i in range(1, VAL_START_DAY)]
    sales_vals = sales[day_cols].values.astype(np.float32)

    # Scale: naive 1-step forecast MSE per series
    diff = np.diff(sales_vals, axis=1)
    scale = np.nanmean(diff ** 2, axis=1)
    scale = np.where(scale == 0, 1e-6, scale)
    scale_series = pd.Series(scale, index=sales["id"].values, name="scale")

    # Revenue weights (sum to 1)
    last28_cols = [f"d_{i}" for i in range(VAL_START_DAY - 28, VAL_START_DAY)]
    last28_sales = sales[["id", "item_id", "store_id"] + last28_cols].copy()
    last_wk = calendar.loc[calendar["d"] == f"d_{VAL_START_DAY - 1}", "wm_yr_wk"].values[0]
    prices_last = prices[prices["wm_yr_wk"] == last_wk][["store_id", "item_id", "sell_price"]]
    last28_sales = last28_sales.merge(prices_last, on=["store_id", "item_id"], how="left")
    last28_sales["sell_price"] = last28_sales["sell_price"].fillna(0)
    revenue = last28_sales[last28_cols].values * last28_sales["sell_price"].values[:, None]
    avg_revenue = revenue.mean(axis=1)
    weights = avg_revenue / (avg_revenue.sum() + 1e-9)
    weight_series = pd.Series(weights, index=sales["id"].values, name="weight")

    return weight_series, scale_series


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: build modelling-ready DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(max_lags: int = 56) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns:
        df        : long-format DataFrame ready for feature engineering in train.py
        sales_wide: original wide sales (needed for WRMSSE denominator)
        weights   : per-series WRMSSE weights
    """
    sales    = load_sales()
    calendar = load_calendar()
    prices   = load_prices()

    weights,scale = compute_wrmsse_weights(sales, prices, calendar)

    df = melt_sales(sales)
    # Keep only days we need (drop very early days to save memory,
    # but keep enough for the largest lag)
    df = df[df["d"] >= (VAL_START_DAY - max_lags - 28)]
    df = merge_calendar(df, calendar)
    df = merge_prices(df, prices)
    df = encode_categoricals(df)
    df = apply_lag_safety_mask(df)

    df["sell_price"] = df["sell_price"].astype(np.float32)
    df.sort_values(["id", "d"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, sales, weights, scale



if __name__ == "__main__":
    df, sales, weights, scale = build_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.dtypes)
    print(f"Weights sample:\n{weights.head()}")
