"""
train.py — Enhanced LightGBM model for M5 Forecasting with target encoding refinement.

THIS FILE IS MODIFIED BY THE AGENT. Each run produces:
  results/preds_{RUN_ID}.parquet
  results/log_{RUN_ID}.json

Hypothesis: TARGET_ENCODING_SMOOTHING — Improve target encoding with smoothing and 
            out-of-fold encoding to reduce overfitting and capture stronger patterns.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from config import (
    VAL_START_DAY, VAL_END_DAY, HORIZON, RESULTS_DIR,
)
from data_prep import build_dataset
from evaluate import wrmsse, load_preds

# ── Run metadata ─────────────────────────────────────────────────────────────
RUN_ID    = os.environ.get("RUN_ID", "target_encoding_smoothing")
HYPOTHESIS = (
    "TARGET_ENCODING_SMOOTHING: Improve target encoding with smoothing and "
    "out-of-fold encoding to reduce overfitting and capture stronger patterns."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=56)

# ── 2. Feature engineering ────────────────────────────────────────────────────
print("[train] Engineering features …")

# ── 2a. Expanded lag features
LAG_DAYS = [7, 14, 21, 28]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Rolling mean features
ROLL_WINDOWS = [7, 28]
for window in ROLL_WINDOWS:
    df[f"rmean_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).mean())
          .astype(np.float32)
    )

# ── 2c. Exponential weighted moving averages
EWM_SPANS = [7, 28]
for span in EWM_SPANS:
    df[f"ewm_{span}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).ewm(span=span, adjust=False).mean())
          .astype(np.float32)
    )

# ── 2d. Expanding window statistics
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
df["expand_min"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(7).expanding(min_periods=1).min())
      .astype(np.float32)
)
df["expand_max"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(7).expanding(min_periods=1).max())
      .astype(np.float32)
)

# ── 2e. Price features
df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

# ── 2f. Price change features
PRICE_CHANGE_WINDOWS = [7, 28]
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
df["price_rank_store_cat"] = (
    df.groupby(["store_id", "cat_id", "d"])["sell_price"]
      .rank(method="min", ascending=False)
      .astype(np.float32)
)

# Price percentiles within groups
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
df["price_pct_dept"] = (
    df.groupby(["dept_id", "d"])["sell_price"]
      .rank(pct=True)
      .astype(np.float32)
)

# ── 2h. Enhanced target encoding with smoothing and out-of-fold (NEW)
TARGET_ENCODING_COLS = ["item_id", "dept_id", "cat_id", "store_id"]
train_mask = df["d"] < VAL_START_DAY
train_data = df[train_mask].copy().reset_index(drop=True)

def smoothed_target_encoding(data, col, target_col, k=100, f=1):
    """
    Compute smoothed target encoding with regularization.
    Formula: (count * mean + k * global_mean) / (count + k)
    """
    global_mean = data[target_col].mean()
    agg = data.groupby(col)[target_col].agg(['mean', 'count']).reset_index()
    agg[f'{col}_smooth_enc'] = (agg['count'] * agg['mean'] + k * global_mean) / (agg['count'] + k)
    return agg[[col, f'{col}_smooth_enc']].set_index(col).to_dict()[f'{col}_smooth_enc']

def out_of_fold_target_encoding(data, col, target_col, n_splits=5):
    """
    Compute out-of-fold target encoding to reduce overfitting.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_encoded = np.zeros(len(data))
    
    global_mean = data[target_col].mean()
    
    for train_idx, val_idx in kf.split(data):
        train_fold = data.iloc[train_idx]
        val_fold = data.iloc[val_idx]
        
        # Create encoding from training fold
        encoding_map = train_fold.groupby(col)[target_col].mean().to_dict()
        
        # Apply to validation fold
        oof_encoded[val_idx] = val_fold[col].map(encoding_map).fillna(global_mean)
    
    return oof_encoded

# Apply enhanced target encoding
for col in TARGET_ENCODING_COLS:
    # Smoothed encoding
    smooth_encoding = smoothed_target_encoding(train_data, col, 'sales', k=50)
    df[f"{col}_smooth_enc"] = df[col].map(smooth_encoding).fillna(train_data['sales'].mean()).astype(np.float32)
    
    # Out-of-fold encoding for training data
    oof_encoding = out_of_fold_target_encoding(train_data, col, 'sales')
    oof_dict = dict(zip(train_data[col], oof_encoding))
    
    # For validation data, use simple mean encoding
    val_encoding = train_data.groupby(col)['sales'].mean().to_dict()
    
    # Combine encodings
    df[f"{col}_oof_enc"] = df[col].map(
        lambda x: oof_dict.get(x) if x in oof_dict else val_encoding.get(x, train_data['sales'].mean())
    ).astype(np.float32)

# Add interaction features between target encodings
df["item_store_enc"] = df["item_id_smooth_enc"] * df["store_id_smooth_enc"]
df["dept_cat_enc"] = df["dept_id_smooth_enc"] * df["cat_id_smooth_enc"]

# ── 3. Define features & split ────────────────────────────────────────────────
FEATURE_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI",
    "sell_price", "price_momentum",
] + [f"lag_{l}" for l in LAG_DAYS] \
  + [f"rmean_{w}" for w in ROLL_WINDOWS] \
  + [f"ewm_{s}" for s in EWM_SPANS] \
  + ["expand_mean", "expand_std", "expand_min", "expand_max"] \
  + [f"price_change_{w}" for w in PRICE_CHANGE_WINDOWS] \
  + [f"price_volatility_{w}" for w in PRICE_CHANGE_WINDOWS] \
  + ["price_rank_store", "price_rank_cat", "price_rank_dept", "price_rank_store_cat"] \
  + ["price_pct_store", "price_pct_cat", "price_pct_dept"] \
  + [f"{col}_smooth_enc" for col in TARGET_ENCODING_COLS] \
  + [f"{col}_oof_enc" for col in TARGET_ENCODING_COLS] \
  + ["item_store_enc", "dept_cat_enc"]

# Training rows: days up to (VAL_START_DAY - 1), drop NaN lags
train_df = df[df["d"] < VAL_START_DAY].dropna(subset=FEATURE_COLS)
# Validation rows: last HORIZON days
val_df   = df[(df["d"] >= VAL_START_DAY) & (df["d"] <= VAL_END_DAY)]

X_train = train_df[FEATURE_COLS]
y_train = train_df["sales"]
X_val   = val_df[FEATURE_COLS]

print(f"[train] Train size: {len(X_train):,}  Val size: {len(X_val):,}")

# ── 4. Train LightGBM ─────────────────────────────────────────────────────────
print("[train] Training LightGBM …")

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
N_ESTIMATORS = 500

dtrain = lgb.Dataset(X_train, label=y_train)
model  = lgb.train(
    params,
    dtrain,
    num_boost_round=N_ESTIMATORS,
    callbacks=[lgb.log_evaluation(100)],
)

# ── 5. Predict validation window ─────────────────────────────────────────────
print("[train] Predicting …")
preds_raw = model.predict(X_val[FEATURE_COLS], num_iteration=model.best_iteration)
val_df = val_df.copy()
val_df["pred"] = np.clip(preds_raw, 0, None).astype(np.float32)

# Build wide predictions
preds_wide = val_df.pivot_table(index="id", columns="d", values="pred").reset_index()
preds_wide.columns = ["id"] + [f"d_{int(c)}" for c in preds_wide.columns[1:]]

# ── 6. Evaluate WRMSSE ───────────────────────────────────────────────────────
day_cols = [f"d_{i}" for i in range(VAL_START_DAY, VAL_END_DAY + 1)]
actuals_wide = sales_wide[["id"] + day_cols].copy()

score = wrmsse(preds_wide, actuals_wide, weights, scale)
duration = time.time() - start_time

# ── 7. Save outputs ───────────────────────────────────────────────────────────
pred_path = RESULTS_DIR / f"preds_{RUN_ID}.parquet"
log_path  = RESULTS_DIR / f"log_{RUN_ID}.json"

preds_wide.to_parquet(pred_path, index=False)

log = {
    "run_id":       RUN_ID,
    "hypothesis":   HYPOTHESIS,
    "wrmsse":       round(score, 6),
    "duration_sec": round(duration, 1),
    "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_features":   len(FEATURE_COLS),
    "n_train_rows": len(X_train),
    "params":       params,
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print(f"WRMSSE: {score:.5f}")