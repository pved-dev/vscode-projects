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
RUN_ID    = os.environ.get("RUN_ID", "lgb_optimized_hyperparams")
HYPOTHESIS = (
    "LGB_OPTIMIZED_HYPERPARAMS: Fine-tune LightGBM hyperparameters for M5 forecasting. "
    "Increase num_leaves to 512 for more complex patterns, reduce learning_rate to 0.03 "
    "with n_estimators=1200 for stable convergence, set min_child_samples=20 for better "
    "leaf quality, increase lambda_l2=0.5 for stronger regularization, and add early_stopping "
    "based on internal validation to prevent overfitting while maintaining the proven feature set."
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

# ── 2b. Comprehensive rolling statistics
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
    
    # Extended rolling stats
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
    df[f"rq25_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).quantile(0.25))
          .astype(np.float32)
    )
    df[f"rq75_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).quantile(0.75))
          .astype(np.float32)
    )

# ── 2c. Expanding window statistics
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

# ── 2d. Price momentum
df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

# ── 2e. Target encoding for key categories
TARGET_ENCODING_COLS = ["item_id", "store_id", "dept_id"]
train_mask = df["d"] < VAL_START_DAY
train_data = df[train_mask]
global_mean = train_data["sales"].mean()

for col in TARGET_ENCODING_COLS:
    encoding_map = train_data.groupby(col)["sales"].mean().to_dict()
    df[f"{col}_enc"] = df[col].map(encoding_map).fillna(global_mean).astype(np.float32)

# ── 2f. Sales trend
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
    "rmean_7", "rmean_14", "rmean_28", 
    "rstd_7", "rstd_14", "rstd_28",
    "rmin_7", "rmin_14", "rmin_28",
    "rmax_7", "rmax_14", "rmax_28",
    "rq25_7", "rq25_14", "rq25_28",
    "rq75_7", "rq75_14", "rq75_28",
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

# ── 4. Train LightGBM with optimized hyperparameters ─────────────────────────
print("[train] Training LightGBM with optimized hyperparameters …")

lgb_params = {
    "objective":        "tweedie",
    "tweedie_variance_power": 1.1,
    "metric":           "rmse",
    "num_leaves":       512,
    "learning_rate":    0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "min_child_samples": 20,
    "lambda_l2":        0.5,
    "verbosity":        -1,
    "n_jobs":           -1,
    "seed":             42,
}
N_ESTIMATORS = 1200

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
        lgb.early_stopping(stopping_rounds=50, verbose=True)
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