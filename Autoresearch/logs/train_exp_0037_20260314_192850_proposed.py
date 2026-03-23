"""
train.py — Enhanced LightGBM model for M5 Forecasting with gradient-based feature selection.

THIS FILE IS MODIFIED BY THE AGENT. Each run produces:
  results/preds_{RUN_ID}.parquet
  results/log_{RUN_ID}.json

Hypothesis: GRADIENT_FEATURE_SELECTION — Use gradient-based feature importance and selection 
            to identify the most predictive features and reduce noise from irrelevant features.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel

from config import (
    VAL_START_DAY, VAL_END_DAY, HORIZON, RESULTS_DIR,
)
from data_prep import build_dataset
from evaluate import wrmsse, load_preds

# ── Run metadata ─────────────────────────────────────────────────────────────
RUN_ID    = os.environ.get("RUN_ID", "gradient_feature_selection")
HYPOTHESIS = (
    "GRADIENT_FEATURE_SELECTION: Use gradient-based feature importance and selection "
    "to identify the most predictive features and reduce noise from irrelevant features."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=56)

# ── 2. Feature engineering ────────────────────────────────────────────────────
print("[train] Engineering features …")

# ── 2a. Expanded lag features
LAG_DAYS = [7, 14, 21, 28, 35, 42]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Rolling mean and std features
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

# ── 2c. Rolling min/max features
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

# ── 2d. Exponential weighted moving averages
EWM_SPANS = [7, 14, 28]
for span in EWM_SPANS:
    df[f"ewm_{span}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).ewm(span=span, adjust=False).mean())
          .astype(np.float32)
    )

# ── 2e. Expanding window statistics
df["expand_mean"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(7).expanding(min_periods=1).mean())
      .astype(np.float32)
)
df["expand_std"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(7).expanding(min_periods=1).std())
      .astype(np.float32)
)

# ── 2f. Price features
df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

# Price change and volatility
PRICE_CHANGE_WINDOWS = [7, 14, 28]
for window in PRICE_CHANGE_WINDOWS:
    df[f"price_change_{window}"] = (
        df.groupby("id")["sell_price"]
          .transform(lambda x: x / x.shift(window).replace(0, np.nan))
          .astype(np.float32)
    )
    df[f"price_volatility_{window}"] = (
        df.groupby("id")["sell_price"]
          .transform(lambda x: x.rolling(window, min_periods=1).std() / x.rolling(window, min_periods=1).mean())
          .astype(np.float32)
    )

# ── 2g. Price ranking features
df["price_rank_store"] = (
    df.groupby(["store_id", "d"])["sell_price"]
      .rank(method="min", ascending=False)
      .astype(np.float32)
)
df["price_rank_cat"] = (
    df.groupby(["cat_id", "d"])["sell_price"]
      .rank(method="min", ascending=False)
      .astype(np.float32)
)
df["price_rank_dept"] = (
    df.groupby(["dept_id", "d"])["sell_price"]
      .rank(method="min", ascending=False)
      .astype(np.float32)
)

# Price percentiles
df["price_pct_store"] = (
    df.groupby(["store_id", "d"])["sell_price"]
      .rank(pct=True)
      .astype(np.float32)
)
df["price_pct_cat"] = (
    df.groupby(["cat_id", "d"])["sell_price"]
      .rank(pct=True)
      .astype(np.float32)
)

# ── 2h. Basic target encoding
TARGET_ENCODING_COLS = ["item_id", "dept_id", "cat_id", "store_id"]
train_mask = df["d"] < VAL_START_DAY
train_data = df[train_mask]
global_mean = train_data["sales"].mean()

for col in TARGET_ENCODING_COLS:
    encoding_map = train_data.groupby(col)["sales"].mean().to_dict()
    df[f"{col}_enc"] = df[col].map(encoding_map).fillna(global_mean).astype(np.float32)

# ── 2i. Sales ratio features
df["sales_to_rmean_7"] = (df["sales"] / (df["rmean_7"] + 1e-8)).astype(np.float32)
df["sales_to_expand_mean"] = (df["sales"] / (df["expand_mean"] + 1e-8)).astype(np.float32)

# ── 2j. Trend features
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

# ── 3. Initial feature set ────────────────────────────────────────────────────
INITIAL_FEATURE_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI",
    "sell_price", "price_momentum",
] + [f"lag_{l}" for l in LAG_DAYS] \
  + [f"rmean_{w}" for w in ROLL_WINDOWS] \
  + [f"rstd_{w}" for w in ROLL_WINDOWS] \
  + [f"rmin_{w}" for w in [7, 28]] \
  + [f"rmax_{w}" for w in [7, 28]] \
  + [f"ewm_{s}" for s in EWM_SPANS] \
  + ["expand_mean", "expand_std"] \
  + [f"price_change_{w}" for w in PRICE_CHANGE_WINDOWS] \
  + [f"price_volatility_{w}" for w in PRICE_CHANGE_WINDOWS] \
  + ["price_rank_store", "price_rank_cat", "price_rank_dept"] \
  + ["price_pct_store", "price_pct_cat"] \
  + [f"{col}_enc" for col in TARGET_ENCODING_COLS] \
  + ["sales_to_rmean_7", "sales_to_expand_mean"] \
  + ["trend_7", "trend_28"]

# Training rows: days up to (VAL_START_DAY - 1), drop NaN lags
train_df = df[df["d"] < VAL_START_DAY].dropna(subset=INITIAL_FEATURE_COLS)
# Validation rows: last HORIZON days
val_df   = df[(df["d"] >= VAL_START_DAY) & (df["d"] <= VAL_END_DAY)]

X_train_initial = train_df[INITIAL_FEATURE_COLS]
y_train = train_df["sales"]
X_val_initial = val_df[INITIAL_FEATURE_COLS]

print(f"[train] Initial feature set size: {len(INITIAL_FEATURE_COLS)}")
print(f"[train] Train size: {len(X_train_initial):,}  Val size: {len(X_val_initial):,}")

# ── 4. Feature selection using LightGBM feature importance ──────────────────
print("[train] Performing gradient-based feature selection …")

# Train initial model for feature importance
params_selection = {
    "objective":        "tweedie",
    "tweedie_variance_power": 1.1,
    "metric":           "rmse",
    "num_leaves":       63,
    "learning_rate":    0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq":     1,
    "min_child_samples": 20,
    "lambda_l2":        0.05,
    "verbosity":        -1,
    "n_jobs":           -1,
    "seed":             42,
}

dtrain_selection = lgb.Dataset(X_train_initial, label=y_train)
model_selection = lgb.train(
    params_selection,
    dtrain_selection,
    num_boost_round=100,
    callbacks=[lgb.log_evaluation(0)],
)

# Get feature importances
feature_importance = model_selection.feature_importance(importance_type='gain')
feature_names = X_train_initial.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Select top features based on cumulative importance
cumsum_importance = importance_df['importance'].cumsum()
total_importance = importance_df['importance'].sum()
threshold = 0.95  # Keep features that contribute to 95% of total importance

selected_indices = cumsum_importance <= (threshold * total_importance)
if selected_indices.sum() < 20:  # Ensure minimum number of features
    selected_indices = importance_df.head(20).index
    
selected_features = importance_df.loc[selected_indices, 'feature'].tolist()

print(f"[train] Selected {len(selected_features)} features out of {len(INITIAL_FEATURE_COLS)}")
print(f"[train] Top 10 features: {selected_features[:10]}")

# ── 5. Train final model with selected features ─────────────────────────────
X_train = X_train_initial[selected_features]
X_val = X_val_initial[selected_features]

print("[train] Training final LightGBM with selected features …")

params = {
    "objective":        "tweedie",
    "tweedie_variance_power": 1.1,
    "metric":           "rmse",
    "num_leaves":       127,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "min_child_samples": 20,
    "lambda_l2":        0.1,
    "verbosity":        -1,
    "n_jobs":           -1,
    "seed":             42,
}
N_ESTIMATORS = 600

dtrain = lgb.Dataset(X_train, label=y_train)
model = lgb.train(
    params,
    dtrain,
    num_boost_round=N_ESTIMATORS,
    callbacks=[lgb.log_evaluation(100)],
)

# ── 6. Predict validation window ─────────────────────────────────────────────
print("[train] Predicting …")
preds_raw = model.predict(X_val, num_iteration=model.best_iteration)
val_df = val_df.copy()
val_df["pred"] = np.clip(preds_raw, 0, None).astype(np.float32)

# Build wide predictions
preds_wide = val_df.pivot_table(index="id", columns="d", values="pred").reset_index()
preds_wide.columns = ["id"] + [f"d_{int(c)}" for c in preds_wide.columns[1:]]

# ── 7. Evaluate WRMSSE ───────────────────────────────────────────────────────
day_cols = [f"d_{i}" for i in range(VAL_START_DAY, VAL_END_DAY + 1)]
actuals_wide = sales_wide[["id"] + day_cols].copy()

score = wrmsse(preds_wide, actuals_wide, weights, scale)
duration = time.time() - start_time

# ── 8. Save outputs ───────────────────────────────────────────────────────────
pred_path = RESULTS_DIR / f"preds_{RUN_ID}.parquet"
log_path  = RESULTS_DIR / f"log_{RUN_ID}.json"

preds_wide.to_parquet(pred_path, index=False)

log = {
    "run_id":       RUN_ID,
    "hypothesis":   HYPOTHESIS,
    "wrmsse":       round(score, 6),
    "duration_sec": round(duration, 1),
    "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_features":   len(selected_features),
    "n_initial_features": len(INITIAL_FEATURE_COLS),
    "n_train_rows": len(X_train),
    "selected_features": selected_features,
    "params":       params,
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print(f"WRMSSE: {score:.5f}")