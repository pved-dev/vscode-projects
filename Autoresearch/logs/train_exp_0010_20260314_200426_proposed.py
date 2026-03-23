"""
train.py — LightGBM/XGBoost model for M5 Forecasting.

THIS FILE IS MODIFIED BY THE AGENT. Each run must:
  1. Save predictions to results/preds_{RUN_ID}.parquet
  2. Print DONE on the last line

DO NOT compute or print WRMSSE — the harness does this independently.
DO NOT modify evaluate.py or data_prep.py.
DO NOT change the validation window (VAL_START_DAY to VAL_END_DAY).
"""
import json
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import VAL_START_DAY, VAL_END_DAY, RESULTS_DIR
from data_prep import build_dataset

RUN_ID     = os.environ.get("RUN_ID", "baseline")
HYPOTHESIS = (
    "BASELINE: Lag features [7, 28], rolling mean windows [7, 28], "
    "LightGBM with default parameters."
)

start_time = time.time()

print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=56)

print("[train] Engineering features …")
for lag in [7, 28]:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

for window in [7, 28]:
    df[f"rmean_{window}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(7).rolling(window, min_periods=1).mean())
          .astype(np.float32)
    )

df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

FEATURE_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI",
    "sell_price", "price_momentum",
    "lag_7", "lag_28", "rmean_7", "rmean_28",
]

train_df = df[df["d"] < VAL_START_DAY].dropna(subset=FEATURE_COLS)
val_df   = df[(df["d"] >= VAL_START_DAY) & (df["d"] <= VAL_END_DAY)].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df["sales"]
X_val   = val_df[FEATURE_COLS]

print(f"[train] Train size: {len(X_train):,}  Val size: {len(X_val):,}")
print("[train] Training LightGBM …")

params = {
    "objective": "tweedie", "tweedie_variance_power": 1.1,
    "metric": "rmse", "num_leaves": 127, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "min_child_samples": 20, "lambda_l1": 0.1, "lambda_l2": 0.1,
    "verbosity": -1, "n_jobs": -1, "seed": 42,
}

model = lgb.train(
    params, lgb.Dataset(X_train, label=y_train),
    num_boost_round=500, callbacks=[lgb.log_evaluation(100)],
)

print("[train] Predicting …")
val_df["pred"] = np.clip(
    model.predict(X_val[FEATURE_COLS], num_iteration=model.best_iteration), 0, None
).astype(np.float32)

preds_wide = val_df.pivot_table(index="id", columns="d", values="pred").reset_index()
preds_wide.columns = ["id"] + [f"d_{int(c)}" for c in preds_wide.columns[1:]]

for col in [f"d_{i}" for i in range(VAL_START_DAY, VAL_END_DAY + 1)]:
    if col not in preds_wide.columns:
        preds_wide[col] = 0.0

pred_path = RESULTS_DIR / f"preds_{RUN_ID}.parquet"
preds_wide.to_parquet(pred_path, index=False)

duration = time.time() - start_time
with open(RESULTS_DIR / f"log_{RUN_ID}.json", "w") as f:
    json.dump({"run_id": RUN_ID, "hypothesis": HYPOTHESIS,
               "duration_sec": round(duration, 1),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print("DONE")