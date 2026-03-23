import json
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    VAL_START_DAY, VAL_END_DAY, HORIZON, RESULTS_DIR,
)
from data_prep import build_dataset

# ── Run metadata ─────────────────────────────────────────────────────────────
RUN_ID    = os.environ.get("RUN_ID", "lgb_baseline_v1")
HYPOTHESIS = (
    "BASELINE_V1: Clean baseline with proven features — lags, rolling stats, "
    "price features, target encoding, and calendar features. "
    "Use tweedie objective, n_estimators=800 with early stopping, "
    "regularized params to stay within time budget."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=42)

# ── 2. Core feature engineering ────────────────────────────────────────────────
print("[train] Engineering features …")

# ── 2a. Lag features
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 35, 42]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Lag difference and ratio features
df["lag_diff_1_7"]   = (df["lag_1"]  - df["lag_7"]).astype(np.float32)
df["lag_diff_7_14"]  = (df["lag_7"]  - df["lag_14"]).astype(np.float32)
df["lag_diff_14_28"] = (df["lag_14"] - df["lag_28"]).astype(np.float32)
df["lag_diff_28_42"] = (df["lag_28"] - df["lag_42"]).astype(np.float32)

df["lag_ratio_1_7"]   = (df["lag_1"]  / (df["lag_7"]  + 0.01)).astype(np.float32)
df["lag_ratio_7_14"]  = (df["lag_7"]  / (df["lag_14"] + 0.01)).astype(np.float32)
df["lag_ratio_14_28"] = (df["lag_14"] / (df["lag_28"] + 0.01)).astype(np.float32)
df["lag_ratio_21_35"] = (df["lag_21"] / (df["lag_35"] + 0.01)).astype(np.float32)

# Weighted lag combinations
df["wlag_short"] = (0.5*df["lag_1"] + 0.3*df["lag_2"] + 0.2*df["lag_3"]).astype(np.float32)
df["wlag_med"]   = (0.4*df["lag_7"] + 0.3*df["lag_14"] + 0.3*df["lag_21"]).astype(np.float32)
df["wlag_long"]  = (0.4*df["lag_28"] + 0.35*df["lag_35"] + 0.25*df["lag_42"]).astype(np.float32)

df["lag_weekly_monthly"] = (
    (df["lag_7"] + df["lag_14"] + df["lag_21"] + df["lag_28"]) / 4
).astype(np.float32)

# ── 2c. Rolling statistics
ROLL_WINDOWS = [7, 14, 28, 56]
for window in ROLL_WINDOWS:
    df[f"rmean_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).mean())
          .astype(np.float32)
    )
    df[f"rstd_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).std())
          .astype(np.float32)
    )

# Rolling min/max
for window in [7, 28]:
    df[f"rmin_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).min())
          .astype(np.float32)
    )
    df[f"rmax_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).max())
          .astype(np.float32)
    )

# ── 2d. Expanding window statistics
df["exp_mean"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(7).expanding(min_periods=1).mean())
      .astype(np.float32)
)
df["exp_std"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(7).expanding(min_periods=1).std())
      .astype(np.float32)
)

# ── 2e. Sales trend
df["trend_7"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1) - x.shift(8))
      .astype(np.float32)
)

# ── 2f. Price features
df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

df["price_rank_store"] = (
    df.groupby(["store_id", "d"])["sell_price"]
      .rank(pct=True)
      .astype(np.float32)
)
df["price_rank_cat"] = (
    df.groupby(["cat_id", "d"])["sell_price"]
      .rank(pct=True)
      .astype(np.float32)
)

cat_price_avg  = df.groupby(["cat_id",  "d"])["sell_price"].transform("mean")
dept_price_avg = df.groupby(["dept_id", "d"])["sell_price"].transform("mean")

df["price_gap_cat"]   = (df["sell_price"] - cat_price_avg).astype(np.float32)
df["price_gap_dept"]  = (df["sell_price"] - dept_price_avg).astype(np.float32)
df["price_ratio_cat"] = (df["sell_price"] / (cat_price_avg  + 0.01)).astype(np.float32)
df["price_ratio_dept"]= (df["sell_price"] / (dept_price_avg + 0.01)).astype(np.float32)

# ── 2g. Target encoding with smoothing
TARGET_ENCODING_COLS = ["item_id", "store_id", "dept_id", "cat_id"]
train_mask  = df["d"] < VAL_START_DAY
train_data  = df[train_mask].copy()
global_mean = train_data["sales"].mean()

SMOOTHING_ALPHA = 50.0

for col in TARGET_ENCODING_COLS:
    stats = train_data.groupby(col).agg(
        group_mean=("sales", "mean"),
        group_count=("sales", "count")
    ).reset_index()
    stats["smoothed_mean"] = (
        (stats["group_count"] * stats["group_mean"] + SMOOTHING_ALPHA * global_mean)
        / (stats["group_count"] + SMOOTHING_ALPHA)
    )
    enc_map = dict(zip(stats[col], stats["smoothed_mean"]))
    df[f"{col}_enc_smooth"] = df[col].map(enc_map).fillna(global_mean).astype(np.float32)

# Item × store target encoding
item_store_stats = train_data.groupby(["item_id", "store_id"]).agg(
    group_mean=("sales", "mean"),
    group_count=("sales", "count")
).reset_index()
item_store_stats["smoothed_mean"] = (
    (item_store_stats["group_count"] * item_store_stats["group_mean"] + SMOOTHING_ALPHA * global_mean)
    / (item_store_stats["group_count"] + SMOOTHING_ALPHA)
)
item_store_map = {
    (r["item_id"], r["store_id"]): r["smoothed_mean"]
    for _, r in item_store_stats.iterrows()
}
df["item_store_enc"] = df.apply(
    lambda r: item_store_map.get((r["item_id"], r["store_id"]), global_mean), axis=1
).astype(np.float32)

# ── 2h. Day-of-week × store interaction encoding
dow_store_stats = train_data.groupby(["wday", "store_id"]).agg(
    group_mean=("sales", "mean"),
    group_count=("sales", "count")
).reset_index()
dow_store_stats["smoothed_mean"] = (
    (dow_store_stats["group_count"] * dow_store_stats["group_mean"] + SMOOTHING_ALPHA * global_mean)
    / (dow_store_stats["group_count"] + SMOOTHING_ALPHA)
)
dow_store_map = {
    (r["wday"], r["store_id"]): r["smoothed_mean"]
    for _, r in dow_store_stats.iterrows()
}
df["dow_store_enc"] = df.apply(
    lambda r: dow_store_map.get((r["wday"], r["store_id"]), global_mean), axis=1
).astype(np.float32)

# ── 3. Feature set ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Identifiers / categoricals
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    # Calendar
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI",
    # Price
    "sell_price", "price_momentum",
    "price_rank_store", "price_rank_cat",
    "price_gap_cat", "price_gap_dept",
    "price_ratio_cat", "price_ratio_dept",
    # Lag features
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28", "lag_35", "lag_42",
    # Lag differences / ratios
    "lag_diff_1_7", "lag_diff_7_14", "lag_diff_14_28", "lag_diff_28_42",
    "lag_ratio_1_7", "lag_ratio_7_14", "lag_ratio_14_28", "lag_ratio_21_35",
    # Weighted lags
    "wlag_short", "wlag_med", "wlag_long", "lag_weekly_monthly",
    # Rolling stats
    "rmean_7", "rmean_14", "rmean_28", "rmean_56",
    "rstd_7",  "rstd_14",  "rstd_28",  "rstd_56",
    "rmin_7",  "rmin_28",
    "rmax_7",  "rmax_28",
    # Expanding stats
    "exp_mean", "exp_std",
    # Trend
    "trend_7",
    # Target encodings
    "item_id_enc_smooth", "store_id_enc_smooth",
    "dept_id_enc_smooth", "cat_id_enc_smooth",
    "item_store_enc", "dow_store_enc",
]

# ── 4. Train / val split ──────────────────────────────────────────────────────
train_df = df[df["d"] < VAL_START_DAY].dropna(subset=FEATURE_COLS)
val_df   = df[(df["d"] >= VAL_START_DAY) & (df["d"] <= VAL_END_DAY)].copy()

# Use last 10% of training for early-stopping validation
cutoff_idx   = int(len(train_df) * 0.90)
train_early  = train_df.iloc[:cutoff_idx]
valid_early  = train_df.iloc[cutoff_idx:]

X_train = train_early[FEATURE_COLS]
y_train = train_early["sales"]
X_valid = valid_early[FEATURE_COLS]
y_valid = valid_early["sales"]
X_val   = val_df[FEATURE_COLS]

print(f"[train] Feature set size: {len(FEATURE_COLS)}")
print(f"[train] Train: {len(X_train):,}  Early-stop valid: {len(X_valid):,}  Val: {len(X_val):,}")

# ── 5. Train LightGBM ─────────────────────────────────────────────────────────
print("[train] Training LightGBM …")

lgb_params = {
    "objective":              "tweedie",
    "tweedie_variance_power": 1.1,
    "metric":                 "rmse",
    "boosting_type":          "gbdt",
    "num_leaves":             128,
    "learning_rate":          0.05,
    "feature_fraction":       0.7,
    "bagging_fraction":       0.7,
    "bagging_freq":           1,
    "min_child_samples":      50,
    "min_gain_to_split":      0.0,
    "lambda_l2":              0.1,
    "lambda_l1":              0.0,
    "verbosity":              -1,
    "n_jobs":                 -1,
    "seed":                   42,
}
N_ESTIMATORS = 800

dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=True)

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=N_ESTIMATORS,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.log_evaluation(50),
        lgb.early_stopping(stopping_rounds=50, verbose=True),
    ],
)

best_iter = lgb_model.best_iteration
print(f"[train] Best iteration: {best_iter}")

# ── 6. Predict & save ─────────────────────────────────────────────────────────
print("[train] Predicting …")
preds = lgb_model.predict(X_val, num_iteration=best_iter)
preds = np.clip(preds, 0, None).astype(np.float32)

val_df["pred"] = preds

preds_wide = val_df.pivot_table(index="id", columns="d", values="pred").reset_index()
preds_wide.columns = ["id"] + [f"d_{int(c)}" for c in preds_wide.columns[1:]]

os.makedirs(RESULTS_DIR, exist_ok=True)
pred_path = RESULTS_DIR / f"preds_{RUN_ID}.parquet"
log_path  = RESULTS_DIR / f"log_{RUN_ID}.json"

preds_wide.to_parquet(pred_path, index=False)

duration = time.time() - start_time

log = {
    "run_id":          RUN_ID,
    "hypothesis":      HYPOTHESIS,
    "wrmsse":          0.0,
    "duration_sec":    round(duration, 1),
    "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_features":      len(FEATURE_COLS),
    "n_train_rows":    len(X_train),
    "lgb_params":      lgb_params,
    "lag_features":    LAG_DAYS,
    "rolling_windows": ROLL_WINDOWS,
    "best_iteration":  best_iter,
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print("DONE")