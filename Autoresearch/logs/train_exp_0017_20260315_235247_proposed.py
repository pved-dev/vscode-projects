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
RUN_ID    = os.environ.get("RUN_ID", "lgb_stronger_reg_v2")
HYPOTHESIS = (
    "Try further strengthened LightGBM regularization: lambda_l2=8.0, lambda_l1=1.0, "
    "min_child_samples=250, min_gain_to_split=0.05, max_depth=7 (reduced from 8). "
    "Run LightGBM only (no XGBoost blend) to isolate effect. "
    "Best single LGB was exp_0007 at 0.94350 — trying to push further with stricter tree constraints."
)

start_time = time.time()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[train] Building dataset …")
df, sales_wide, weights, scale = build_dataset(max_lags=365)

# ── 2. Core feature engineering ────────────────────────────────────────────────
print("[train] Engineering features …")

# ── 2a. Lag features (including long lags)
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 35, 42, 49, 56, 84, 168, 365]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

# ── 2b. Zero sale streak (vectorized, no loops)
df["sales_prev"] = df.groupby("id")["sales"].shift(1)
df["nz_flag"] = (df["sales_prev"] > 0).astype(np.int32)
df["nz_cumsum"] = df.groupby("id")["nz_flag"].cumsum()
df["row_num"] = df.groupby("id").cumcount()
df["last_nz_row"] = np.where(df["nz_flag"] == 1, df["row_num"], np.nan)
df["last_nz_row"] = (
    df.groupby("id")["last_nz_row"]
      .transform(lambda x: x.ffill().fillna(-1))
)
df["zero_sale_streak"] = (df["row_num"] - df["last_nz_row"]).clip(lower=0).astype(np.float32)
df.drop(columns=["sales_prev", "nz_flag", "nz_cumsum", "row_num", "last_nz_row"], inplace=True)

# ── 2c. Lag differences and ratios
df["lag_diff_7_14"]   = (df["lag_7"]   - df["lag_14"]).astype(np.float32)
df["lag_diff_14_28"]  = (df["lag_14"]  - df["lag_28"]).astype(np.float32)
df["lag_diff_1_7"]    = (df["lag_1"]   - df["lag_7"]).astype(np.float32)
df["lag_diff_28_56"]  = (df["lag_28"]  - df["lag_56"]).astype(np.float32)
df["lag_diff_56_168"] = (df["lag_56"]  - df["lag_168"]).astype(np.float32)

df["lag_ratio_7_14"]    = (df["lag_7"]   / (df["lag_14"]  + 0.01)).astype(np.float32)
df["lag_ratio_14_28"]   = (df["lag_14"]  / (df["lag_28"]  + 0.01)).astype(np.float32)
df["lag_ratio_1_7"]     = (df["lag_1"]   / (df["lag_7"]   + 0.01)).astype(np.float32)
df["lag_ratio_28_56"]   = (df["lag_28"]  / (df["lag_56"]  + 0.01)).astype(np.float32)
df["lag_ratio_56_168"]  = (df["lag_56"]  / (df["lag_168"] + 0.01)).astype(np.float32)
df["lag_ratio_168_365"] = (df["lag_168"] / (df["lag_365"] + 0.01)).astype(np.float32)

df["wlag_short"] = (0.5 * df["lag_1"] + 0.3 * df["lag_2"] + 0.2 * df["lag_3"]).astype(np.float32)
df["wlag_med"]   = (0.4 * df["lag_7"] + 0.3 * df["lag_14"] + 0.3 * df["lag_21"]).astype(np.float32)
df["wlag_long"]  = (0.4 * df["lag_28"] + 0.35 * df["lag_42"] + 0.25 * df["lag_56"]).astype(np.float32)
df["wlag_vlong"] = (0.5 * df["lag_168"] + 0.5 * df["lag_365"]).astype(np.float32)

# ── 2d. Rolling statistics (shift by 1 to avoid leakage)
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

# ── 2e. Rolling min/max
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

# ── 2f. Expanding window statistics
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

# ── 2g. EWMA features ─────────────────────────────────────────────────────────
EWMA_SPANS = [7, 14, 28, 56]
for span in EWMA_SPANS:
    df[f"ewma_{span}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
          .astype(np.float32)
    )

for span in [7, 28]:
    df[f"ewma_std_{span}"] = (
        df.groupby("id")["sales"]
          .transform(lambda x: x.shift(1).ewm(span=span, min_periods=2).std())
          .fillna(0)
          .astype(np.float32)
    )

df["ewma_momentum_7_28"]  = (df["ewma_7"]  / (df["ewma_28"]  + 0.01)).astype(np.float32)
df["ewma_momentum_7_56"]  = (df["ewma_7"]  / (df["ewma_56"]  + 0.01)).astype(np.float32)
df["ewma_momentum_14_56"] = (df["ewma_14"] / (df["ewma_56"]  + 0.01)).astype(np.float32)
df["ewma_momentum_28_56"] = (df["ewma_28"] / (df["ewma_56"]  + 0.01)).astype(np.float32)

# ── 2h. Day-of-week seasonal rolling means ────────────────────────────────────
same_wday_lags_4 = ["lag_7", "lag_14", "lag_21", "lag_28"]
same_wday_lags_8 = ["lag_7", "lag_14", "lag_21", "lag_28", "lag_35", "lag_42", "lag_49", "lag_56"]

df["dow_roll_mean_4w"] = (
    df[same_wday_lags_4].mean(axis=1, skipna=True).astype(np.float32)
)
df["dow_roll_mean_8w"] = (
    df[same_wday_lags_8].mean(axis=1, skipna=True).astype(np.float32)
)
df["dow_roll_std_4w"] = (
    df[same_wday_lags_4].std(axis=1, skipna=True).fillna(0).astype(np.float32)
)
df["dow_trend_4w_8w"] = (df["dow_roll_mean_4w"] - df["dow_roll_mean_8w"]).astype(np.float32)
df["dow_ratio_lag7_mean4w"] = (df["lag_7"] / (df["dow_roll_mean_4w"] + 0.01)).astype(np.float32)

# ── 2i. Price features
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

# ── 2j. Target encoding with smoothing
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

# ── 2k. Sales trend
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

# ── 2l. SNAP interaction features
df["snap_any"] = (df["snap_CA"] | df["snap_TX"] | df["snap_WI"]).astype(np.float32)
df["snap_price"] = (df["snap_any"] * df["sell_price"]).astype(np.float32)

# ── 2m. Event interaction features
df["event_price"] = (df["event_flag"] * df["sell_price"]).astype(np.float32)
df["event_lag7"]  = (df["event_flag"] * df["lag_7"].fillna(0)).astype(np.float32)

# ── 2n. Ratio of short-term to long-term rolling mean (momentum signal)
df["rmean_ratio_7_28"]  = (df["rmean_7"]  / (df["rmean_28"] + 0.01)).astype(np.float32)
df["rmean_ratio_7_56"]  = (df["rmean_7"]  / (df["rmean_56"] + 0.01)).astype(np.float32)
df["rmean_ratio_28_56"] = (df["rmean_28"] / (df["rmean_56"] + 0.01)).astype(np.float32)

# ── 2o. Yearly seasonality features using lag_365 and lag_168
df["yearly_ratio"]    = (df["lag_7"]    / (df["lag_365"] + 0.01)).astype(np.float32)
df["half_year_ratio"] = (df["lag_7"]    / (df["lag_168"] + 0.01)).astype(np.float32)
df["trend_vs_year"]   = (df["rmean_28"] / (df["lag_365"] + 0.01)).astype(np.float32)

# ── 3. Feature set ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Categorical identifiers
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    # Calendar
    "wday", "month", "year",
    "event_flag", "event2_flag",
    "snap_CA", "snap_TX", "snap_WI", "snap_any",
    # Price
    "sell_price", "price_momentum",
    "price_rank_store", "price_rank_cat",
    "price_gap_cat", "price_gap_dept",
    "price_ratio_cat", "price_ratio_dept",
    "snap_price", "event_price", "event_lag7",
    "price_roll_mean_7", "price_roll_mean_28",
    "price_roll_std_7", "price_roll_std_28",
    "price_change_7", "price_change_28",
    "price_at_min_28", "price_norm_28",
    # Lags (including long lags)
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21",
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
    "lag_84", "lag_168", "lag_365",
    # Lag combinations
    "lag_diff_7_14", "lag_diff_14_28", "lag_diff_1_7", "lag_diff_28_56",
    "lag_diff_56_168",
    "lag_ratio_7_14", "lag_ratio_14_28", "lag_ratio_1_7", "lag_ratio_28_56",
    "lag_ratio_56_168", "lag_ratio_168_365",
    "wlag_short", "wlag_med", "wlag_long", "wlag_vlong",
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
    # EWMA features
    "ewma_7", "ewma_14", "ewma_28", "ewma_56",
    "ewma_std_7", "ewma_std_28",
    "ewma_momentum_7_28", "ewma_momentum_7_56",
    "ewma_momentum_14_56", "ewma_momentum_28_56",
    # Day-of-week seasonal features
    "dow_roll_mean_4w", "dow_roll_mean_8w",
    "dow_roll_std_4w", "dow_trend_4w_8w",
    "dow_ratio_lag7_mean4w",
    # Yearly seasonality features
    "yearly_ratio", "half_year_ratio", "trend_vs_year",
    # Target encodings
    "item_id_enc_smooth", "store_id_enc_smooth",
    "dept_id_enc_smooth", "cat_id_enc_smooth",
    # Zero sale streak
    "zero_sale_streak",
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

# ── 4. Train LightGBM with further strengthened regularization ───────────────
print("[train] Training LightGBM with further strengthened regularization …")

lgb_params = {
    "objective":              "tweedie",
    "tweedie_variance_power": 1.5,
    "metric":                 "rmse",
    "boosting_type":          "gbdt",
    "num_leaves":             256,
    "max_depth":              7,          # reduced from 8 to 7
    "learning_rate":          0.04,
    "feature_fraction":       0.6,
    "bagging_fraction":       0.7,
    "bagging_freq":           1,
    "min_child_samples":      250,        # increased from 200 to 250
    "min_gain_to_split":      0.05,       # increased from 0.01 to 0.05
    "lambda_l2":              8.0,        # increased from 5.0 to 8.0
    "lambda_l1":              1.0,        # increased from 0.5 to 1.0
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

lgb_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_preds = np.clip(lgb_preds, 0, None).astype(np.float32)
print(f"[train] LightGBM best iteration: {lgb_model.best_iteration}")

# ── 5. Use LightGBM predictions only ─────────────────────────────────────────
print("[train] Using LightGBM-only predictions …")
final_preds = lgb_preds

val_df = val_df.copy()
val_df["pred"] = final_preds

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
    "run_id":               RUN_ID,
    "hypothesis":           HYPOTHESIS,
    "wrmsse":               0.0,
    "duration_sec":         round(duration, 1),
    "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_features":           len(FEATURE_COLS),
    "n_train_rows":         len(X_train),
    "lgb_params":           lgb_params,
    "lag_features":         LAG_DAYS,
    "rolling_windows":      ROLL_WINDOWS,
    "lgb_best_iteration":   lgb_model.best_iteration,
    "blend_weights":        {"lgb": 1.0, "xgb": 0.0},
}
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"[train] Run '{RUN_ID}' complete in {duration:.1f}s")
print("DONE")