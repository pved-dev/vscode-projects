"""
evaluate.py — Clean WRMSSE implementation for M5.
"""
import numpy as np
import pandas as pd
from config import VAL_START_DAY, VAL_END_DAY


def wrmsse(preds, actuals, weights, scale) -> float:
    day_cols = [f"d_{i}" for i in range(VAL_START_DAY, VAL_END_DAY + 1)]

    # Align both on id
    pred_df   = preds.set_index("id")[day_cols]
    actual_df = actuals.set_index("id")[day_cols]

    # Only score ids present in both
    common_ids = pred_df.index.intersection(actual_df.index)
    pred_vals   = pred_df.loc[common_ids].values.astype(np.float64)
    actual_vals = actual_df.loc[common_ids].values.astype(np.float64)
    pred_vals   = np.clip(pred_vals, 0, None)

    # Per-series MSE
    mse = np.mean((pred_vals - actual_vals) ** 2, axis=1)

    # Align weights and scale
    w = weights.reindex(common_ids).values.astype(np.float64)
    s = scale.reindex(common_ids).values.astype(np.float64)

    # Zero out bad series
    w = np.where(np.isfinite(w), w, 0.0)
    s = np.where((np.isfinite(s)) & (s > 0), s, np.nan)

    # WRMSSE
    rmsse   = np.sqrt(mse / s)
    valid   = np.isfinite(rmsse)
    w_total = w[valid].sum()
    score   = (w[valid] * rmsse[valid]).sum() / (w_total + 1e-9)
    return float(score)


def load_preds(path: str) -> pd.DataFrame:
    preds_long = pd.read_parquet(path)
    if "pred" in preds_long.columns:
        preds_wide = preds_long.pivot_table(
            index="id", columns="d", values="pred"
        ).reset_index()
        preds_wide.columns = ["id"] + [f"d_{int(c)}" for c in preds_wide.columns[1:]]
    else:
        preds_wide = preds_long
    return preds_wide