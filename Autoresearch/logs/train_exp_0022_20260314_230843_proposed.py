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
RUN_ID    = os.environ.get("RUN_ID", "ensemble_lgb_xgb")
HYPOTHESIS = (
    "ENSEMBLE_LGB_XGB: Create an ensemble by training both LightGBM and XGBoost "
    "models and blending their predictions with equal weights. Different gradient "
    "boosting implementations may capture complementary patterns, potentially "
    "improving overall forecasting performance."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=56)

# ── 2. Core feature engineering ────────────────────────────────────────────────
print("[train] Engineering core features …")

# ── 2a. Essential lag features
LAG_DAYS = [7, 14, 28]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Key rolling statistics
ROLL_WINDOWS = [7, 28]
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

# ── 2c. Expanding window features
df["expand_mean"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
      .astype(np.float32)
)
df["expand_std"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(1).expanding(min_periods=1).std())
      .astype(np.float32)
)

# ── 2d. Price momentum
df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

# ── 2e. Price change features
df["price_change_7"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(7).replace(0, np.nan))
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

# ── 2h. Sales ratio to recent average
df["sales_to_rmean_7"] = (df["lag_7"] / (df["rmean_7"] + 1e-8)).astype(np.float32)

# ── 3. Feature set ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI",
    "sell_price", "price_momentum", "price_change_7",
    "lag_7", "lag_14", "lag_28",
    "rmean_7", "rmean_28", "rstd_7", "rstd_28",
    "expand_mean", "expand_std",
    "trend_7", "sales_to_rmean_7",
    "item_id_enc", "store_id_enc", "dept_id_enc"
]

# Training rows: days up to (VAL_START_DAY - 1), drop NaN lags
train_df = df[df["d"] < VAL_START_DAY].dropna(subset=FEATURE_COLS)
# Validation rows: last HORIZON days
val_df   = df[(df["d"] >= VAL_START_DAY) & (df["d"] <= VAL_END_DAY)]

X_train = train_df[FEATURE_COLS]
y_train = train_df["sales"]
X_val = val_df[FEATURE_COLS]

print(f"[train] Feature set size: {len(FEATURE_COLS)}")
print(f"[train] Train size: {len(X_train):,}  Val size: {len(X_val):,}")

# ── 4. Train LightGBM ─────────────────────────────────────────────────────────
print("[train] Training LightGBM …")

lgb_params = {
    "objective":        "tweedie",
    "tweedie_variance_power": 1.1,
    "metric":           "rmse",
    "num_leaves":       255,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "min_child_samples": 50,
    "lambda_l2":        0.1,
    "verbosity":        -1,
    "n_jobs":           -1,
    "seed":             42,
}
N_ESTIMATORS = 800

dtrain = lgb.Dataset(X_train, label=y_train)
lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=N_ESTIMATORS,
    callbacks=[lgb.log_evaluation(100)],
)

# ── 5. Train XGBoost ──────────────────────────────────────────────────────────
print("[train] Training XGBoost …")
import xgboost as xgb

xgb_params = {
    "objective":        "reg:tweedie",
    "tweedie_variance_power": 1.1,
    "eval_metric":      "rmse",
    "max_depth":        8,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "lambda":           0.1,
    "verbosity":        0,
    "n_jobs":           -1,
    "seed":             42,
}

dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
xgb_model = xgb.train(
    xgb_params,
    dtrain_xgb,
    num_boost_round=N_ESTIMATORS,
    verbose_eval=100,
)

# ── 6. Predict validation window with ensemble ──────────────────────────────
print("[train] Predicting with ensemble …")
dval_xgb = xgb.DMatrix(X_val)

lgb_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
xgb_preds = xgb_model.predict(dval_xgb)

# Ensemble with equal weights
ensemble_preds = 0.5 * lgb_preds + 0.5 * xgb_preds

val_df = val_df.copy()
val_df["pred"] = np.clip(ensemble_preds, 0, None).astype(np.float32)

# Build wide predictions
preds_wide = val_df.pivot_table(index="id", columns="d", values="pred").reset_index()
preds_wide.columns = ["id"] + [f"d_{int(c)}" for c in preds_wide.columns[1:]]

# ── 7. Save outputs ───────────────────────────────────────────────────────────
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
    "xgb_params":   xgb_params,
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print("DONE")