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
RUN_ID    = os.environ.get("RUN_ID", "lgb_ewma_features_v2")
HYPOTHESIS = (
    "Add exponentially weighted moving average (EWMA) features with multiple spans "
    "(7, 14, 28, 56) as additional lag-style features. EWMA gives more weight to recent "
    "observations while retaining long-term memory — complementary to rolling means. "
    "Also add EWMA-based momentum (ratio of short to long EWMA) and EWMA std. "
    "All other settings unchanged from exp_0006 (best so far)."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=56)

# ── 2. Core feature engineering ────────────────────────────────────────────────
print("[train] Engineering features …")

# ── 2a. Lag features
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Lag differences and ratios
df["lag_diff_7_14"]  = (df["lag_7"]  - df["lag_14"]).astype(np.float32)
df["lag_diff_14_28"] = (df["lag_14"] - df["lag_28"]).astype(np.float32)
df["lag_diff_1_7"]   = (df["lag_1"]  - df["lag_7"]).astype(np.float32)
df["lag_diff_28_56"] = (df["lag_28"] - df["lag_56"]).astype(np.float32)

df["lag_ratio_7_14"]  = (df["lag_7"]  / (df["lag_14"] + 0.01)).astype(np.float32)
df["lag_ratio_14_28"] = (df["lag_14"] / (df["lag_28"] + 0.01)).astype(np.float32)
df["lag_ratio_1_7"]   = (df["lag_1"]  / (df["lag_7"]  + 0.01)).astype(np.float32)
df["lag_ratio_28_56"] = (df["lag_28"] / (df["lag_56"] + 0.01)).astype(np.float32)

df["wlag_short"] = (0.5 * df["lag_1"] + 0.3 * df["lag_2"] + 0.2 * df["lag_3"]).astype(np.float32)
df["wlag_med"]   = (0.4 * df["lag_7"] + 0.3 * df["lag_14"] + 0.3 * df["lag_21"]).astype(np.float32)
df["wlag_long"]  = (0.4 * df["lag_28"] + 0.35 * df["lag_42"] + 0.25 * df["lag_56"]).astype(np.float32)

# ── 2c. Rolling statistics (shift by 1 to avoid leakage)
ROLL_WINDOWS = [7, 14, 28, 56]
for window in ROLL_WINDOWS:
    df[f"rmean_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
          .astype(np.float32)
    )
    df[f"rstd_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
          .astype(np.float32)
    )

# ── 2d. Rolling min/max
for window in [7, 28, 56]:
    df[f"rmin_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
          .astype(np.float32)
    )
    df[f"rmax_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
          .astype(np.float32)
    )

# ── 2e. Expanding window statistics
df["exp_mean"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
      .astype(np.float32)
)
df["exp_std"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1).expanding(min_periods=1).std())
      .astype(np.float32)
)

# ── 2f. EWMA features (NEW) ───────────────────────────────────────────────────
# Exponentially weighted moving average — shift by 1 to avoid leakage
# Spans chosen to match short/medium/long-term patterns
EWMA_SPANS = [7, 14, 28, 56]
for span in EWMA_SPANS:
    df[f"ewma_{span}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
          .astype(np.float32)
    )

# EWMA standard deviation (volatility signal)
for span in [7, 28]:
    df[f"ewma_std_{span}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).ewm(span=span, min_periods=2).std())
          .fillna(0)
          .astype(np.float32)
    )

# EWMA momentum: ratio of short-span to long-span EWMA
df["ewma_momentum_7_28"]  = (df["ewma_7"]  / (df["ewma_28"]  + 0.01)).astype(np.float32)
df["ewma_momentum_7_56"]  = (df["ewma_7"]  / (df["ewma_56"]  + 0.01)).astype(np.float32)
df["ewma_momentum_14_56"] = (df["ewma_14"] / (df["ewma_56"]  + 0.01)).astype(np.float32)
df["ewma_momentum_28_56"] = (df["ewma_28"] / (df["ewma_56"]  + 0.01)).astype(np.float32)

# ── 2g. Price features (existing)
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

df["price_gap_cat"]    = (df["sell_price"] - cat_price_avg).astype(np.float32)
df["price_gap_dept"]   = (df["sell_price"] - dept_price_avg).astype(np.float32)
df["price_ratio_cat"]  = (df["sell_price"] / (cat_price_avg  + 0.01)).astype(np.float32)
df["price_ratio_dept"] = (df["sell_price"] / (dept_price_avg + 0.01)).astype(np.float32)

# Price history features
df["price_roll_mean_7"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
      .astype(np.float32)
)
df["price_roll_mean_28"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x.shift(1).rolling(28, min_periods=1).mean())
      .astype(np.float32)
)
df["price_roll_std_7"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x.shift(1).rolling(7, min_periods=2).std())
      .fillna(0)
      .astype(np.float32)
)
df["price_roll_std_28"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x.shift(1).rolling(28, min_periods=2).std())
      .fillna(0)
      .astype(np.float32)
)
df["price_change_7"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x - x.shift(7))
      .astype(np.float32)
)
df["price_change_28"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x - x.shift(28))
      .astype(np.float32)
)
df["price_roll_min_28"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x.shift(1).rolling(28, min_periods=1).min())
      .astype(np.float32)
)
df["price_at_min_28"] = (
    (df["sell_price"] <= df["price_roll_min_28"] + 0.001).astype(np.float32)
)
df["price_norm_28"] = (
    df["sell_price"] / (df["price_roll_mean_28"] + 0.01)
).astype(np.float32)

# ── 2h. Target encoding with smoothing
TARGET_ENCODING_COLS = ["item_id", "store_id", "dept_id", "cat_id"]
train_mask  = df["d"] < VAL_START_DAY
train_data  = df[train_mask].copy()
global_mean = train_data["sales"].mean()

SMOOTHING_ALPHA = 50.0

for col in TARGET_ENCODING_COLS:
    full_stats = train_data.groupby(col).agg(
        group_mean=("sales", "mean"),
        group_count=("sales", "count")
    ).reset_index()
    full_stats["smoothed_mean"] = (
        (full_stats["group_count"] * full_stats["group_mean"] + SMOOTHING_ALPHA * global_mean) /
        (full_stats["group_count"] + SMOOTHING_ALPHA)
    )
    enc_map = dict(zip(full_stats[col], full_stats["smoothed_mean"]))
    df[f"{col}_enc_smooth"] = df[col].map(enc_map).fillna(global_mean).astype(np.float32)

# ── 2i. Sales trend
df["trend_7"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1) - x.shift(8))
      .astype(np.float32)
)
df["trend_28"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1) - x.shift(29))
      .astype(np.float32)
)

# ── 2j. SNAP interaction features
df["snap_any"] = (df["snap_CA"] | df["snap_TX"] | df["snap_WI"]).astype(np.float32)
df["snap_price"] = (df["snap_any"] * df["sell_price"]).astype(np.float32)

# ── 2k. Event interaction features
df["event_price"] = (df["event_flag"] * df["sell_price"]).astype(np.float32)
df["event_lag7"]  = (df["event_flag"] * df["lag_7"].fillna(0)).astype(np.float32)

# ── 2l. Ratio of short-term to long-term rolling mean (momentum signal)
df["rmean_ratio_7_28"]  = (df["rmean_7"]  / (df["rmean_28"] + 0.01)).astype(np.float32)
df["rmean_ratio_7_56"]  = (df["rmean_7"]  / (df["rmean_56"] + 0.01)).astype(np.float32)
df["rmean_ratio_28_56"] = (df["rmean_28"] / (df["rmean_56"] + 0.01)).astype(np.float32)

# ── 3. Feature set ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Categorical identifiers
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    # Calendar
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI", "snap_any",
    # Price (existing)
    "sell_price", "price_momentum",
    "price_rank_store", "price_rank_cat",
    "price_gap_cat", "price_gap_dept",
    "price_ratio_cat", "price_ratio_dept",
    "snap_price", "event_price", "event_lag7",
    # Price history features
    "price_roll_mean_7", "price_roll_mean_28",
    "price_roll_std_7", "price_roll_std_28",
    "price_change_7", "price_change_28",
    "price_at_min_28", "price_norm_28",
    # Lags
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21",
    "lag_28", "lag_35", "lag_42", "lag_56",
    # Lag combinations
    "lag_diff_7_14", "lag_diff_14_28", "lag_diff_1_7", "lag_diff_28_56",
    "lag_ratio_7_14", "lag_ratio_14_28", "lag_ratio_1_7", "lag_ratio_28_56",
    "wlag_short", "wlag_med", "wlag_long",
    # Rolling statistics
    "rmean_7", "rmean_14", "rmean_28", "rmean_56",
    "rstd_7",  "rstd_14",  "rstd_28",  "rstd_56",
    "rmin_7",  "rmin_28",  "rmin_56",
    "rmax_7",  "rmax_28",  "rmax_56",
    # Rolling ratios
    "rmean_ratio_7_28", "rmean_ratio_7_56", "rmean_ratio_28_56",
    # Expanding
    "exp_mean", "exp_std",
    # Trend
    "trend_7", "trend_28",
    # EWMA features (NEW)
    "ewma_7", "ewma_14", "ewma_28", "ewma_56",
    "ewma_std_7", "ewma_std_28",
    "ewma_momentum_7_28", "ewma_momentum_7_56",
    "ewma_momentum_14_56", "ewma_momentum_28_56",
    # Target encodings
    "item_id_enc_smooth", "store_id_enc_smooth",
    "dept_id_enc_smooth", "cat_id_enc_smooth",
]

# Training rows: days up to (VAL_START_DAY - 1), drop NaN lags
train_df = df[df["d"] < VAL_START_DAY].dropna(subset=FEATURE_COLS)
# Validation rows: last HORIZON days
val_df   = df[(df["d"] >= VAL_START_DAY) & (df["d"] <= VAL_END_DAY)]

# Split training data for early stopping — use last 10% of training time as validation
cutoff_day = int(train_df["d"].quantile(0.90))
train_early = train_df[train_df["d"] <= cutoff_day]
valid_early = train_df[train_df["d"] >  cutoff_day]

X_train = train_early[FEATURE_COLS]
y_train = train_early["sales"]
X_valid = valid_early[FEATURE_COLS]
y_valid = valid_early["sales"]
X_val   = val_df[FEATURE_COLS]

print(f"[train] Feature set size: {len(FEATURE_COLS)}")
print(f"[train] Train size: {len(X_train):,}  Early valid size: {len(X_valid):,}  Final val size: {len(X_val):,}")

# ── 4. Train LightGBM ─────────────────────────────────────────────────────────
print("[train] Training LightGBM …")

lgb_params = {
    "objective":              "tweedie",
    "tweedie_variance_power": 1.5,
    "metric":                 "rmse",
    "boosting_type":          "gbdt",
    "num_leaves":             64,
    "learning_rate":          0.05,
    "feature_fraction":       0.6,
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

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

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

# ── 5. Predict validation window ──────────────────────────────────────────────
print("[train] Predicting …")

preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
preds = np.clip(preds, 0, None).astype(np.float32)

val_df = val_df.copy()
val_df["pred"] = preds

# Build wide predictions
preds_wide = val_df.pivot_table(index="id", columns="d", values="pred").reset_index()
preds_wide.columns = ["id"] + [f"d_{int(c)}" for c in preds_wide.columns[1:]]

# ── 6. Save outputs ───────────────────────────────────────────────────────────
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
    "best_iteration":  lgb_model.best_iteration,
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print("DONE")