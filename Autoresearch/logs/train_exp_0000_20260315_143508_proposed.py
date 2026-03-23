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
RUN_ID    = os.environ.get("RUN_ID", "lgb_dart_ensemble")
HYPOTHESIS = (
    "LGB_DART_ENSEMBLE: Switch to DART booster with dropout regularization for improved "
    "generalization. DART prevents overfitting by randomly dropping trees during training, "
    "similar to dropout in neural networks. Use moderate dropout rates (0.1-0.2) with "
    "aggressive early stopping to maintain stability while improving out-of-sample performance "
    "on the temporal validation window."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=42)

# ── 2. Core feature engineering ────────────────────────────────────────────────
print("[train] Engineering features …")

# ── 2a. Extended lag features (from best experiment)
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 35, 42]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Advanced lag combinations (from best experiment)
df["lag_diff_7_14"] = (df["lag_7"] - df["lag_14"]).astype(np.float32)
df["lag_diff_14_28"] = (df["lag_14"] - df["lag_28"]).astype(np.float32)
df["lag_diff_1_7"] = (df["lag_1"] - df["lag_7"]).astype(np.float32)
df["lag_diff_28_42"] = (df["lag_28"] - df["lag_42"]).astype(np.float32)

df["lag_ratio_7_14"] = (df["lag_7"] / (df["lag_14"] + 0.01)).astype(np.float32)
df["lag_ratio_14_28"] = (df["lag_14"] / (df["lag_28"] + 0.01)).astype(np.float32)
df["lag_ratio_1_7"] = (df["lag_1"] / (df["lag_7"] + 0.01)).astype(np.float32)
df["lag_ratio_21_35"] = (df["lag_21"] / (df["lag_35"] + 0.01)).astype(np.float32)

df["wlag_short"] = (0.5 * df["lag_1"] + 0.3 * df["lag_2"] + 0.2 * df["lag_3"]).astype(np.float32)
df["wlag_med"] = (0.4 * df["lag_7"] + 0.3 * df["lag_14"] + 0.3 * df["lag_21"]).astype(np.float32)
df["wlag_long"] = (0.4 * df["lag_28"] + 0.35 * df["lag_35"] + 0.25 * df["lag_42"]).astype(np.float32)

df["lag_7_28_interaction"] = (df["lag_7"] * df["lag_28"]).astype(np.float32)
df["lag_14_28_interaction"] = (df["lag_14"] * df["lag_28"]).astype(np.float32)
df["lag_weekly_monthly"] = (df["lag_7"] + df["lag_14"] + df["lag_21"] + df["lag_28"]).astype(np.float32) / 4

# ── 2c. Advanced price features (keep best ones)
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

# Price gap features relative to category averages
cat_price_avg = df.groupby(["cat_id", "d"])["sell_price"].transform("mean")
dept_price_avg = df.groupby(["dept_id", "d"])["sell_price"].transform("mean")

df["price_gap_cat"] = (df["sell_price"] - cat_price_avg).astype(np.float32)
df["price_gap_dept"] = (df["sell_price"] - dept_price_avg).astype(np.float32)

df["price_ratio_cat"] = (df["sell_price"] / (cat_price_avg + 0.01)).astype(np.float32)
df["price_ratio_dept"] = (df["sell_price"] / (dept_price_avg + 0.01)).astype(np.float32)

# ── 2d. Rolling statistics (keep proven ones)
ROLL_WINDOWS = [7, 14, 28]
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

# ── 2e. Expanding window statistics
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

# ── 2f. Basic price momentum (keep)
df["price_momentum"] = (
    df.groupby("id")["sell_price"]
      .transform(lambda x: x / x.shift(1).replace(0, np.nan))
      .astype(np.float32)
)

# ── 2g. Target encoding with smoothing (keep from best model)
TARGET_ENCODING_COLS = ["item_id", "store_id", "dept_id", "cat_id"]
train_mask = df["d"] < VAL_START_DAY
train_data = df[train_mask].copy()
global_mean = train_data["sales"].mean()

SMOOTHING_ALPHA = 50.0
N_FOLDS = 5
train_data["fold"] = np.random.randint(0, N_FOLDS, len(train_data))

for col in TARGET_ENCODING_COLS:
    df[f"{col}_enc_smooth"] = np.nan
    
    for fold in range(N_FOLDS):
        train_fold = train_data[train_data["fold"] != fold]
        val_fold = train_data[train_data["fold"] == fold]
        
        encoding_stats = train_fold.groupby(col).agg({
            "sales": ["mean", "count"]
        }).reset_index()
        encoding_stats.columns = [col, "group_mean", "group_count"]
        
        encoding_stats["smoothed_mean"] = (
            (encoding_stats["group_count"] * encoding_stats["group_mean"] + SMOOTHING_ALPHA * global_mean) /
            (encoding_stats["group_count"] + SMOOTHING_ALPHA)
        )
        
        encoding_map = dict(zip(encoding_stats[col], encoding_stats["smoothed_mean"]))
        
        fold_mask = df["d"].isin(val_fold["d"]) & df["id"].isin(val_fold["id"])
        df.loc[fold_mask, f"{col}_enc_smooth"] = df.loc[fold_mask, col].map(encoding_map)
    
    full_encoding_stats = train_data.groupby(col).agg({
        "sales": ["mean", "count"]
    }).reset_index()
    full_encoding_stats.columns = [col, "group_mean", "group_count"]
    full_encoding_stats["smoothed_mean"] = (
        (full_encoding_stats["group_count"] * full_encoding_stats["group_mean"] + SMOOTHING_ALPHA * global_mean) /
        (full_encoding_stats["group_count"] + SMOOTHING_ALPHA)
    )
    full_encoding_map = dict(zip(full_encoding_stats[col], full_encoding_stats["smoothed_mean"]))
    
    df[f"{col}_enc_smooth"] = df[f"{col}_enc_smooth"].fillna(
        df[col].map(full_encoding_map).fillna(global_mean)
    ).astype(np.float32)

# ── 2h. Temporal Consistency Features (keep best from previous)
print("[train] Adding temporal consistency features …")

# Day-of-month effects
df["day_of_month"] = ((df["d"] - 1) % 365) % 28 + 1
df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 28).astype(np.float32)
df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 28).astype(np.float32)

# Weekly seasonality strength
df["weekly_seasonality_strength"] = 0.1  # Default value

for item_id in df["id"].unique()[:500]:  # Reduced for efficiency
    item_mask = df["id"] == item_id
    if item_mask.sum() > 56:
        item_data = df.loc[item_mask, "sales"].values
        weekly_pattern = []
        for i in range(7):
            weekly_sales = item_data[i::7]
            weekly_pattern.append(np.mean(weekly_sales) if len(weekly_sales) > 0 else 0)
        
        weekly_strength = np.std(weekly_pattern) / (np.mean(weekly_pattern) + 0.01)
        df.loc[item_mask, "weekly_seasonality_strength"] = weekly_strength

df["weekly_seasonality_strength"] = df["weekly_seasonality_strength"].fillna(
    df.groupby("cat_id")["weekly_seasonality_strength"].transform("mean")
).fillna(0.1).astype(np.float32)

# Monthly trend component
df["monthly_trend"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: x.shift(28).rolling(28, min_periods=7).mean() - x.shift(56).rolling(28, min_periods=7).mean())
      .astype(np.float32)
)

# Temporal deviation from expected pattern
df["dow_expected"] = (
    df.groupby(["id", "wday"])["sales"]
      .transform(lambda x: x.shift(7).expanding(min_periods=1).mean())
      .astype(np.float32)
)
df["temporal_deviation"] = (df["sales"] - df["dow_expected"]).astype(np.float32)
df["temporal_deviation_abs"] = np.abs(df["temporal_deviation"]).astype(np.float32)

# Seasonal consistency measures
df["seasonal_consistency_7"] = (
    df.groupby("id")["sales"]
      .transform(lambda x: 1 / (1 + (x - x.shift(7)).abs().rolling(7, min_periods=1).mean()))
      .astype(np.float32)
)

# ── 2i. Sales trend (keep)
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
    "price_rank_store", "price_rank_cat",
    "price_gap_cat", "price_gap_dept",
    "price_ratio_cat", "price_ratio_dept",
    "rmean_7", "rmean_14", "rmean_28", 
    "rstd_7", "rstd_14", "rstd_28",
    "exp_mean", "exp_std",
    "trend_7",
    "item_id_enc_smooth", "store_id_enc_smooth", "dept_id_enc_smooth", "cat_id_enc_smooth",
    "day_of_month_sin", "day_of_month_cos",
    "weekly_seasonality_strength",
    "monthly_trend",
    "temporal_deviation", "temporal_deviation_abs",
    "seasonal_consistency_7"
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

# ── 4. Train LightGBM with DART booster ─────────────────────────────────────
print("[train] Training LightGBM with DART booster …")

lgb_params = {
    "objective":        "tweedie",
    "tweedie_variance_power": 1.1,
    "metric":           "rmse",
    "boosting_type":    "dart",  # NEW: Switch to DART
    "drop_rate":        0.15,    # NEW: Moderate dropout rate
    "max_drop":         50,      # NEW: Maximum trees to drop
    "skip_drop":        0.5,     # NEW: Probability to skip dropout
    "uniform_drop":     False,   # NEW: Use standard dropout
    "xgboost_dart_mode": False,  # NEW: LightGBM DART mode
    "num_leaves":       64,
    "learning_rate":    0.025,   # Slightly higher for DART
    "feature_fraction": 0.7,     # Slightly higher for DART
    "bagging_fraction": 0.7,     # Slightly higher for DART
    "bagging_freq":     1,
    "min_child_samples": 150,    # Reduced for DART
    "min_gain_to_split": 0.5,    # Reduced for DART
    "lambda_l2":        1.0,     # Reduced since DART provides regularization
    "lambda_l1":        0.05,    # Reduced since DART provides regularization
    "verbosity":        -1,
    "n_jobs":           -1,
    "seed":             42,
}
N_ESTIMATORS = 1000  # Reduced since DART typically needs fewer rounds

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=N_ESTIMATORS,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.log_evaluation(50),   # More frequent logging for DART
        lgb.early_stopping(stopping_rounds=75, verbose=True)  # More aggressive early stopping
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
    "booster_type": "dart",
    "dart_params": {
        "drop_rate": 0.15,
        "max_drop": 50,
        "skip_drop": 0.5
    }
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print("DONE")