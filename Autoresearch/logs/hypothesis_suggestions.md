# Hypothesis Suggestions — AutoResearch M5
Generated from 103 experiments (7 improvements)
Current best WRMSSE: 0.94290 (exp_0069_20260316_042754)

---

# Analysis of Experiment History

## What Produced Improvements
1. **Strong regularization** (exp_0009): lambda_l2=8, lambda_l1=1, min_child_samples=250, max_depth=7
2. **LGB+XGB blending** (exp_0003, 0007, 0052, 0069): 50/50 or 60/40 blend
3. **Training window blending** (exp_0033): 60% all-data model + 40% recent-data model
4. **LGB+XGB with tuned params** (exp_0069): current best 0.94290

## Exhausted Directions
- Tweedie power variations (1.2, 1.3, 1.5 all tried)
- num_leaves variations (255, 256, 320, 384, 512)
- L1/L2 regularization tuning (extensively tried)
- Blend weight tuning (dozens of attempts, marginal gains)
- Price quantile features (no gain)
- SNAP interaction features (no gain)
- Zero-sale streak interactions (no gain)
- Seed ensembles (mostly failed/no gain)

## High-Value Untried Directions (from top M5 solutions)
- **Recursive prediction with proper leakage-free features** (hierarchical models)
- **Item/store-level categorical embeddings via target encoding**
- **Prophet-style decomposition features** (trend + seasonality)
- **Quantile/hierarchical loss functions**
- **Cross-level aggregation features** (dept, category, store aggregates)
- **Day-of-month features and month-end effects**
- **Log-transform target** instead of Tweedie
- **Rolling autocorrelation features**
- **Separate models per aggregation level**
- **Neural network blend** (LSTM/MLP)

---

RANK: 1
HYPOTHESIS: Switch from Tweedie objective to Poisson objective (reg:count:poisson for XGBoost, poisson for LightGBM) combined with log1p-transformed target, since M5 sales are count data and Poisson with log1p target often outperforms Tweedie in practice.
RATIONALE: The current best uses Tweedie power=1.5 but multiple Tweedie power experiments (1.2, 1.3, 1.5) show diminishing returns. Top M5 solutions frequently use Poisson regression or Poisson+log1p target transformation. Poisson is a special case of Tweedie (power=1) and is specifically designed for count data. Log1p transformation of the target before training with Poisson objective makes the model more sensitive to low-sales items which dominate WRMSSE weights.
IMPLEMENTATION: `lgb_params['objective'] = 'poisson'`, `lgb_params['poisson_max_delta_step'] = 0.7`; transform target: `y_train = np.log1p(y_train)`, predictions: `pred = np.expm1(model.predict(X))`; for XGBoost: `xgb_params['objective'] = 'count:poisson'`, `xgb_params['max_delta_step'] = 0.7`. Keep all other best params from exp_0069.
EXPECTED_GAIN: 0.005-0.015
PRIORITY: HIGH

---

RANK: 2
HYPOTHESIS: Add hierarchical aggregation features: rolling means and lags computed at dept_id×store_id, cat_id×store_id, and dept_id levels (cross-item aggregates), which capture department-level trends that individual item models miss.
RATIONALE: Top M5 solutions consistently use aggregated features at multiple hierarchy levels. Currently features are computed per item-store. Adding dept-store level lag_7/14/28 rolling means captures systematic department shocks (e.g., entire category going on promotion). These features provide strong signal especially for slow-moving items where individual item lags are noisy. This is a completely untried feature direction in the experiment history.
IMPLEMENTATION: `df['dept_store_rmean_28'] = df.groupby(['dept_id','store_id','wday'])['sales'].transform(lambda x: x.shift(1).rolling(28).mean())`; similarly compute `cat_store_rmean_28`, `dept_store_lag_7`, `dept_store_lag_28`, `cat_store_lag_7`. Add these ~8 new columns to feature set. Use best params from exp_0069.
EXPECTED_GAIN: 0.005-0.012
PRIORITY: HIGH

---

RANK: 3
HYPOTHESIS: Add calendar-based cyclical features (day_of_month sine/cosine encoding, month sine/cosine, week_of_year sine/cosine) plus month-end/month-start binary flags, as M5 data shows strong within-month sales patterns tied to paycheck cycles.
RATIONALE: Currently only wday (day of week) is used as a calendar feature. Top M5 solutions use richer time encoding. Walmart sales show strong month-end effects (SNAP benefits, paycheck timing). Day-of-month encoded as sin/cos captures smooth cyclical patterns. Week-of-year captures seasonal patterns. None of these have been tried in 103 experiments. The M5 competition dataset spans 5+ years, making these patterns learnable.
IMPLEMENTATION: `df['dom'] = df['date'].dt.day`; `df['dom_sin'] = np.sin(2*np.pi*df['dom']/31)`; `df['dom_cos'] = np.cos(2*np.pi*df['dom']/31)`; `df['month_sin'] = np.sin(2*np.pi*df['month']/12)`; `df['month_cos'] = np.cos(2*np.pi*df['month']/12)`; `df['woy_sin'] = np.sin(2*np.pi*df['date'].dt.isocalendar().week/52)`; `df['is_month_end'] = (df['dom'] >= 28).astype(int)`; `df['is_month_start'] = (df['dom'] <= 3).astype(int)`. Add to feature list in LGB+XGB blend.
EXPECTED_GAIN: 0.003-0.008
PRIORITY: HIGH

---

RANK: 4
HYPOTHESIS: Train separate LightGBM models per store (10 stores) with shared feature engineering but store-specific hyperparameters, then aggregate predictions, since different stores have fundamentally different sales distributions and the WRMSSE weights are store-specific.
RATIONALE: All 103 experiments train a single global model. Top M5 solutions often use store-level or state-level models. With 10 Walmart stores having distinct demographics, product mixes, and SNAP eligibility, a single model may underfit store-specific patterns. Store-specific models can better capture local trends. WRMSSE penalizes errors differently by store based on sales volume weights, so improving store-level accuracy directly targets the metric.
IMPLEMENTATION: `stores = df['store_id'].unique()`; for each store: `train_store = train[train['store_id']==store]`; train separate LGB model with same hyperparams as exp_0069 best; `preds[store_mask] = model_store.predict(X_test[store_mask])`; aggregate all store predictions. Use same feature set. Optionally blend with global model (0.3 global + 0.7 per-store).
EXPECTED_GAIN: 0.004-0.010
PRIORITY: HIGH

---

RANK: 5
HYPOTHESIS: Add lag autocorrelation features: rolling correlation between lag_7 and lag_14, rolling coefficient of variation (std/mean) over 28 days, and the ratio of recent mean (lag_7 rolling 4w) to long-term mean (lag_365 rolling 4w), capturing trend and volatility signals.
RATIONALE: The current feature set has many levels of rolling means but no explicit volatility or trend-ratio features. Top M5 solutions use CoV and trend-ratio features to distinguish stable vs volatile items. The ratio of recent-to-historical mean directly captures whether an item is trending up/down vs seasonal baseline. CoV captures intermittency. These are distinct from existing features (which are levels, not ratios/variance). None tried in 103 experiments.
IMPLEMENTATION: `df['rmean_7'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(1).rolling(7).mean())`; `df['rmstd_28'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(1).rolling(28).std())`; `df['cov_28'] = df['rmstd_28'] / (df['rmean_28'] + 1e-3)`; `df['trend_ratio'] = df['rmean_28'] / (df.groupby('id')['sales'].transform(lambda x: x.shift(365).rolling(28).mean()) + 1e-3)`; `df['lag7_14_ratio'] = (df['lag_7'] + 1) / (df['lag_14'] + 1)`. Add to feature set with exp_0069 params.
EXPECTED_GAIN: 0.003-0.008
PRIORITY: HIGH

---

RANK: 6
HYPOTHESIS: Apply WRMSSE-aware sample weighting during training by assigning higher sample weights to items/series with higher WRMSSE weights (based on denominator of WRMSSE), making the model focus more on high-impact series.
RATIONALE: The WRMSSE metric weights series by their historical sales variance and scale. A model trained with uniform sample weights will treat a $10 item the same as a $0.50 item. Top M5 solutions use importance weighting. Items with high revenue × high variance contribute most to WRMSSE. By passing `sample_weight` proportional to item's WRMSSE denominator weight to LightGBM/XGBoost, the model directly optimizes what the metric penalizes. This is a principled improvement not tried in any of 103 experiments.
IMPLEMENTATION: Compute `item_weights = (price * scale_factor)` from M5 evaluation weights CSV; `sample_weight = train_df['id'].map(item_weights_dict)`; normalize: `sample_weight = sample_weight / sample_weight.mean()`; `lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weight)`; same for XGBoost `xgb.DMatrix(..., weight=sample_weight)`. Use exp_0069 hyperparams. Clip weights to [0.1, 10] to prevent extreme influence.
EXPECTED_GAIN: 0.005-0.015
PRIORITY: HIGH

---

RANK: 7
HYPOTHESIS: Add event lead/lag features by creating binary columns for N days before and after each event type (sporting, cultural, national, religious), since sales spikes often occur 1-3 days before events (pre-shopping) and dip 1-2 days after.
RATIONALE: exp_0016 (days_to_next_event) FAILED due to implementation bugs, but the underlying idea is sound. Top M5 winning solutions consistently use event lead/lag features. Rather than computing distance (which caused failures), create simple binary flags: `event_in_3days`, `event_in_2days`, `event_in_1day`, `event_yesterday`, `event_2days_ago`. These are simple vectorized joins on the calendar table, avoiding the loop-based failures of exp_0016.
IMPLEMENTATION: `calendar['event_flag'] = (calendar['event_name_1'].notna()).astype(int)`; create shifted versions: `for d in [1,2,3]: calendar[f'event_lead_{d}'] = calendar['event_flag'].shift(-d).fillna(0)`; `for d in [1,2]: calendar[f'event_lag_{d}'] = calendar['event_flag'].shift(d).fillna(0)`; merge into training df by 'd' column. Also separate by event_type_1 (sporting/cultural/national/religious). Add ~10 binary columns. Use exp_0069 params.
EXPECTED_GAIN: 0.003-0.007
PRIORITY: MEDIUM

---

RANK: 8
HYPOTHESIS: Use a 3-model ensemble: LightGBM trained on all data + LightGBM trained on last 2 years + XGBoost trained on all data, with optimized weights (e.g., 0.4/0.3/0.3), combining the proven training-window diversity from exp_0033 with the LGB+XGB diversity from exp_0052.
RATIONALE: exp_0033 showed that blending all-data + recent-data LGB models improved to 0.94304. exp_0052 showed LGB+XGB blend improved to 0.94291. The current best exp_0069 combines both. However, exp_0033's training-window blending was never combined with XGBoost in a 3-way blend. Adding a recent-window LGB model to the current best LGB+XGB blend should provide orthogonal diversity signal, as the recent model captures recent distribution shifts that both the all-data models may miss.
IMPLEMENTATION: `model_A = lgb.train(params, lgb.Dataset(X_all, y_all))  # all data`; `model_B = lgb.train(params, lgb.Dataset(X_recent, y_recent))  # last 730 days`; `model_C = xgb.train(xgb_params, xgb.DMatrix(X_all, y_all))  # XGBoost all data`; `pred = 0.40 * model_A.predict(X_test) + 0.30 * model_B.predict(X_test) + 0.30 * model_C.predict(X_test)`. Use exp_0069 hyperparams for A and C, exp_0033 window cutoff (day >= 1550) for B.
EXPECTED_GAIN: 0.002-0.006
PRIORITY: MEDIUM

---

RANK: 9
HYPOTHESIS: Add price-momentum features: price change direction (price increased/decreased vs last week), number of consecutive weeks at same price (price_stability), and discount depth (current_price / max_price_52w - 1), capturing promotional pricing signals.
RATIONALE: Current price features include relative price but not price momentum or promotional depth. Top M5 solutions use price-change direction and discount depth as strong signals. A price decrease of 20%+ from 52-week high strongly predicts sales spike. Price stability (consecutive weeks at same price) indicates non-promotional baseline. These features are distinct from price_quantile_52w (tried but showed no gain) because they capture the direction and magnitude of price changes rather than percentile rank. All 103 experiments lack these momentum features.
IMPLEMENTATION: `df['price_lag_7'] = df.groupby('id')['sell_price'].shift(7)  # using wday alignment`; `df['price_change'] = (df['sell_price'] - df['price_lag_7']) / (df['price_lag_7'] + 1e-3)`; `df['price_increased'] = (df['price_change'] > 0.001).astype(int)`; `df['price_decreased'] = (df['price_change'] < -0.001).astype(int)`; `df['discount_depth'] = (df['price_max_52w'] - df['sell_price']) / (df['price_max_52w'] + 1e-3)  # price_max_52w = rolling 365-day max`; add to feature set with exp_0069 params.
EXPECTED_GAIN: 0.002-0.006
PRIORITY: MEDIUM

---

RANK: 10
HYPOTHESIS: Replace the current single-pass recursive prediction with a direct multi-step strategy by training 28 separate LightGBM models, one for each forecast horizon (h=1 to h=28), each using only features available at prediction time for that specific horizon.
RATIONALE: The current approach likely uses recursive/single-model prediction where lag features for days 1-28 are filled iteratively (or imputed), causing error accumulation. Top M5 solutions use DIRECT multi-horizon forecasting where model_h predicts day t+h directly using only known features at time t. This eliminates error propagation in lag features. While computationally expensive (28 models), each model is trained on a subset, and the 28

---

## How to use these suggestions
1. Pick the hypotheses marked HIGH priority
2. Add them to the `## Next Priority` section in `prompt.md`
3. The agent will try them in the next run
4. After each run, re-run this tool to get updated suggestions

## Exhausted directions (do not retry)
-  Refine LGB+XGB blend from exp_0069 (WRMSSE=0.94290
-  Add 52-week rolling price quantile (price_quantile_52w
-  Refine blend weights from exp_0033 (best=0.94304 with 0.6/0.4
-  Refine blend from exp_0033 (best=0.94304 with 0.6/0.4 blend
-  Refine LGB+XGB blend from exp_0052 (WRMSSE=0.94291
-  Refine LGB+XGB blend from exp_0052 (best=0.94291
-  Train 3 LightGBM models with different seeds (42, 123, 777
-  Use best known params: lambda_l1=0.5 (best single model at 0.94333
