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
RUN_ID    = os.environ.get("RUN_ID", "lgb_advanced_lag_combinations")
HYPOTHESIS = (
    "LGB_ADVANCED_LAG_COMBINATIONS: Add sophisticated lag combination features to capture "
    "complex temporal patterns. Include lag differences (lag_7 - lag_14, lag_14 - lag_28) "
    "to detect trend changes, lag ratios (lag_7/lag_14, lag_14/lag_28) for momentum, "
    "weighted lag averages with decay, and seasonal lag interactions (lag_7 * lag_28) "
    "to better model weekly and monthly cyclical patterns in retail sales data."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=42)

# ── 2. Core feature engineering ────────────────────────────────────────────────
print("[train] Engineering features …")

# ── 2a. Extended lag features (from previous best experiment)
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 35, 42]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Advanced lag combinations
# Lag differences to detect trend changes
df["lag_diff_7_14"] = (df["lag_7"] - df["lag_14"]).astype(np.float32)
df["lag_diff_14_28"] = (df["lag_14"] - df["lag_28"]).astype(np.float32)
df["lag_diff_1_7"] = (df["lag_1"] - df["lag_7"]).astype(np.float32)
df["lag_diff_28_42"] = (df["lag_28"] - df["lag_42"]).astype(np.float32)

# Lag ratios for momentum detection
df["lag_ratio_7_14"] = (df["lag_7"] / (df["lag_14"] + 0.01)).astype(np.float32)
df["lag_ratio_14_28"] = (df["lag_14"] / (df["lag_28"] + 0.01)).astype(np.float32)
df["lag_ratio_1_7"] = (df["lag_1"] / (df["lag_7"] + 0.01)).astype(np.float32)
df["lag_ratio_21_35"] = (df["lag_21"] / (df["lag_35"] + 0.01)).astype(np.float32)

# Weighted lag averages with exponential decay
df["wlag_short"] = (0.5 * df["lag_1"] + 0.3 * df["lag_2"] + 0.2 * df["lag_3"]).astype(np.float32)
df["wlag_med"] = (0.4 * df["lag_7"] + 0.3 * df["lag_14"] + 0.3 * df["lag_21"]).astype(np.float32)
df["wlag_long"] = (0.4 * df["lag_28"] + 0.35 * df["lag_35"] + 0.25 * df["lag_42"]).astype(np.float32)

# Seasonal lag interactions
df["lag_7_28_interaction"] = (df["lag_7"] * df["lag_28"]).astype(np.float32)
df["lag_14_28_interaction"] = (df["lag_14"] * df["lag_28"]).astype(np.float32)
df["lag_weekly_monthly"] = (df["lag_7"] + df["lag_14"] + df["lag_21"] + df["lag_28"]).astype(np.float32) / 4

# ── 2c. Comprehensive rolling statistics
ROLL_WINDOWS = [7, 14, 28]
for window in ROLL_WINDOWS:
    # Basic stats (keep from previous)
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

# ── 2e. Price momentum
df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

# ── 2f. Target encoding for key categories
TARGET_ENCODING_COLS = ["item_id", "store_id", "dept_id"]
train_mask = df["d"] < VAL_START_DAY
train_data = df[train_mask]
global_mean = train_data["sales"].mean()

for col in TARGET_ENCODING_COLS:
    encoding_map = train_data.groupby(col)["sales"].mean().to_dict()
    df[f"{col}_enc"] = df[col].map(encoding_map).fillna(global_mean).astype(np.float32)

# ── 2g. Sales trend
df["trend_7"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1) - x.shift(8))
      .astype(np.float32)
)

# ── 3. Feature set ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI",
    "sell_price", "price_momentum",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28", "lag_35", "lag_42",
    "lag_diff_7_14", "lag_diff_14_28", "lag_diff_1_7", "lag_diff_28_42",
    "lag_ratio_7_14", "lag_ratio_14_28", "lag_ratio_1_7", "lag_ratio_21_35",
    "wlag_short", "wlag_med", "wlag_long",
    "lag_7_28_interaction", "lag_14_28_interaction", "lag_weekly_monthly",
    "rmean_7", "rmean_14", "rmean_28", 
    "rstd_7", "rstd_14", "rstd_28",
    "exp_mean", "exp_std",
    "trend_7",
    "item_id_enc", "store_id_enc", "dept_id_enc"
]

# Training rows: days up to (VAL_START_DAY - 1), drop NaN lags
train_df = df[df["d"] < VAL_START_DAY].dropna(subset=FEATURE_COLS)
# Validation rows: last HORIZON days
val_df   = df[(df["d"] >= VAL_START_DAY) & (df["d"] <= VAL_END_DAY)]

# Split training data for early stopping validation
train_cutoff = int(0.85 * len(train_df))
train_early = train_df.iloc[:train_cutoff]
valid_early = train_df.iloc[train_cutoff:]

X_train = train_early[FEATURE_COLS]
y_train = train_early["sales"]
X_valid = valid_early[FEATURE_COLS]
y_valid = valid_early["sales"]
X_val = val_df[FEATURE_COLS]

print(f"[train] Feature set size: {len(FEATURE_COLS)}")
print(f"[train] Train size: {len(X_train):,}  Early valid size: {len(X_valid):,}  Final val size: {len(X_val):,}")

# ── 4. Train LightGBM with extreme regularization ───────────────────────────
print("[train] Training LightGBM with extreme regularization …")

lgb_params = {
    "objective":        "tweedie",
    "tweedie_variance_power": 1.1,
    "metric":           "rmse",
    "boosting_type":    "gbdt",
    "num_leaves":       64,
    "learning_rate":    0.02,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq":     1,
    "min_child_samples": 200,
    "min_gain_to_split": 1.0,
    "lambda_l2":        2.0,
    "lambda_l1":        0.1,
    "verbosity":        -1,
    "n_jobs":           -1,
    "seed":             42,
}
N_ESTIMATORS = 1500

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=N_ESTIMATORS,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.log_evaluation(100),
        lgb.early_stopping(stopping_rounds=100, verbose=True)
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
pred_path = RESULTS_DIR / f"preds_{RUN_ID}.parquet"
log_path  = RESULTS_DIR / f"log_{RUN_ID}.json"

preds_wide.to_parquet(pred_path, index=False)

duration = time.time() - start_time

log = {
    "run_id":       RUN_ID,
    "hypothesis":   HYPOTHESIS,
    "wrmsse":       0.0,
    "duration_sec": round(duration, 1),
    "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_features":   len(FEATURE_COLS),
    "n_train_rows": len(X_train),
    "lgb_params":   lgb_params,
    "lag_features": LAG_DAYS,
    "rolling_windows": ROLL_WINDOWS,
    "best_iteration": lgb_model.best_iteration,
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print("DONE")