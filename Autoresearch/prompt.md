# AutoResearch M5 — Research Directive

## Task
You are an autonomous ML research agent. Your job is to iteratively improve a
LightGBM/XGBoost forecasting model on the **Kaggle M5 Forecasting competition**
dataset. You will read `train.py`, study the experiment log, propose ONE focused
hypothesis, rewrite `train.py` to test it, and stop. A separate harness will run
the script, evaluate it, and call you again.

## Competition Context
- **Goal**: Forecast unit sales for 3,049 Walmart products across 10 stores in 3
  US states (CA, TX, WI) for a 28-day horizon.
- **Metric**: WRMSSE (Weighted Root Mean Squared Scaled Error) — lower is better.
  This weights errors by product volume and price, so high-revenue items matter more.
- **Data**: `sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`
- **Validation**: Last 28 days of training data (days 1886–1913).

## Lag Feature Safety — CRITICAL
The data pipeline in `data_prep.py` automatically masks unsafe lag features to
NaN for validation days where they would cause leakage. This means:
- You MAY use any lag including lag_1 through lag_27
- For validation day 1886, lag_1 uses day 1885 (safe — last training day)
- For validation day 1887, lag_1 uses day 1886 (unsafe — automatically masked to NaN)
- The masking is handled for you — just include the lag in your feature list
- DO NOT manually filter or drop lags — the pipeline handles it


## Next Priority — Try These First (Ranked by Expected Gain)

### HIGH PRIORITY (try immediately)
- **Poisson objective + log1p target**: change objective to 'poisson', transform 
  target with np.log1p(), predictions with np.expm1(). Keep exp_0069 params.
  Implementation: `lgb_params['objective']='poisson'`, `lgb_params['poisson_max_delta_step']=0.7`

- **WRMSSE sample weights**: pass item weights to LightGBM/XGBoost training.
  `lgb.Dataset(X_train, y_train, weight=sample_weight)` where sample_weight 
  comes from _weights Series already computed in agent.py. Clip to [0.1, 10].

- **Calendar cyclical encoding**: add day_of_month sin/cos, month sin/cos, 
  is_month_end (dom>=28), is_month_start (dom<=3). Fully vectorized via date column.

- **Event lead/lag binary flags**: simple calendar merge — event_lead_1/2/3 days, 
  event_lag_1/2 days. No loops, just shifted event_flag column on calendar df.

### MEDIUM PRIORITY (try after high priority exhausted)
- **3-model ensemble**: LGB all-data + LGB recent-data (day>=1550) + XGB, 
  blend 0.40/0.30/0.30. Combines exp_0033 window diversity with exp_0069 LGB+XGB.

- **Price momentum features**: price_change direction (up/down vs last week), 
  discount_depth = (price_max_52w - sell_price) / price_max_52w

## Proven Dead Ends — DO NOT RETRY
These have been tested multiple times and always fail or hurt the score:
- Zero-sale row downsampling (tried 3 times — always worse)
- DART booster (times out repeatedly)
- Separate models per department (slower, no improvement)
- Time-based cross-validation (too slow for time budget)
- item_id × store_id target encoding (tried 8+ times — no improvement)
- Any further target encoding variations — direction exhausted
- Aggregated rolling mean features at dept/store/cat level (always times out)
- days_since_release feature (tried 5+ times — always times out or crashes)
- price drop flag binary feature (tried 3+ times — no improvement beyond 0.944)

## Hypothesis Space — Other Areas to Explore
### Feature Engineering
- Long lags: lag_56, lag_84, lag_168, lag_365
- Same-weekday rolling means (lag_7, lag_14, lag_21, lag_28 combinations)
- Zero sale streak: consecutive days of zero sales (use groupby + cumsum)
- Days to next event: precompute on calendar df then merge — no loops

### Model Architecture
- Switch between LightGBM and XGBoost (or blend them)
- Tune num_leaves, max_depth, min_child_samples
- Adjust learning_rate + n_estimators
- Regularization: lambda_l1, lambda_l2, min_gain_to_split

### Training Strategy
- Sample weighting by WRMSSE weights
- Early stopping with a held-out training window
- Tweedie vs Poisson vs regression objective
- Feature selection based on importance scores

### Post-processing
- Clip extreme outliers by category
- Exponential smoothing blend on final predictions

## Stagnation Rule — Mandatory
Look at the last 5 experiments in the log. If ALL of them are within 0.002 
WRMSSE of each other and NONE improved on the best score, you are in a local 
optimum. You MUST:
1. Stop all blend ratio tuning, regularization tweaking, and seed ensembling
2. Pick something from the Next Priority list that has NOT been tried
3. If everything in Next Priority has been tried, pick a completely new 
   feature engineering direction not seen in any recent experiment

## Constraints
- **Time budget**: {TIME_BUDGET_MINUTES} minutes per experiment — hard limit
- **Vectorized operations ONLY**: Never use Python loops over items, series, or days.
  Fast patterns:
  - zero_sale_streak: use groupby + cumsum, not loops
  - days_to_event: precompute on calendar df then merge
  - Any feature looping over 30,490 items will always timeout
- **n_estimators**: Keep at or below 800. Use early_stopping_rounds=50
- When running LGB + XGB blend, use num_boost_round=400 for each model maximum
- Total training rounds across all models must not exceed 800
- **Memory**: Stay under 16 GB RAM
- **One change at a time**: Single isolated hypothesis only
- **Do not change**: `evaluate.py`, `data_prep.py`, `config.py`
- **Do not compute WRMSSE**: Harness evaluates independently. Save predictions and print DONE.

## Output Contract
Return ONLY the full runnable contents of `train.py` — no preamble, no markdown
fences, no explanation. The file must:
1. Save predictions to `results/preds_{RUN_ID}.parquet`
2. Save a JSON log to `results/log_{RUN_ID}.json`
3. Print `DONE` on the last line

## Experiment Log (most recent first)
{EXPERIMENT_LOG}

## Current `train.py`
{CURRENT_TRAIN_PY}