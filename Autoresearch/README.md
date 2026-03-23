# AutoResearch M5

An autonomous ML research loop inspired by Andrej Karpathy's *autoresearch* project,
applied to the **Kaggle M5 Forecasting** competition using LightGBM and XGBoost.

The core idea: **you iterate on the research directive (`prompt.md`);
the AI agent iterates on the training code (`train.py`).**

---

## Architecture

```
prompt.md          ← YOU edit this (research goals, allowed changes)
    │
    ▼
agent.py           ← Reads prompt + experiment log → calls Claude API
    │                 Writes new train.py → runs it → logs result
    │                 Reverts or commits based on WRMSSE
    ▼
train.py           ← AGENT edits this every iteration
    │
    ├── data_prep.py   (stable — loads & melts M5 data, computes WRMSSE weights)
    ├── evaluate.py    (stable — WRMSSE metric)
    └── config.py      (stable — paths, time budget, API settings)
```

```
Each iteration:
  [Claude API] → new train.py
       ↓
  [run train.py with TIME_BUDGET_MINUTES limit]
       ↓
  Parse WRMSSE from stdout
       ↓
  Improvement?  ──YES──► save as train_best.py, git commit, update log
       │
      NO
       │
       └──► revert to train_best.py, log failure
```

---

## Quickstart

### 1. Prerequisites

```bash
# Python 3.10+
export ANTHROPIC_API_KEY=sk-ant-...
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_kaggle_api_key
```

### 2. Setup

```bash
chmod +x run.sh
./run.sh setup
```

### 3. Download M5 Data

```bash
./run.sh data
# Downloads to ./data/
# Files needed:
#   sales_train_validation.csv
#   calendar.csv
#   sell_prices.csv
#   sample_submission.csv
```

### 4. Smoke Test (one manual run)

```bash
./run.sh once
```

### 5. Start the Agent Loop

```bash
./run.sh loop
# Runs up to MAX_EXPERIMENTS=200 iterations autonomously
# Each capped at TIME_BUDGET_MINUTES=5
# ~12 experiments/hour → ~100 while you sleep
```

### 6. Check Progress

```bash
./run.sh report
```

---

## M5 Competition Details

| Property | Value |
|---|---|
| Dataset | Walmart unit sales |
| Series | 3,049 products × 10 stores = 30,490 series |
| Granularity | Daily |
| Forecast horizon | 28 days |
| Train period | Days 1–1,913 |
| Validation | Days 1,886–1,913 (last 28 days) |
| Metric | **WRMSSE** (Weighted Root Mean Squared Scaled Error) |

### WRMSSE

Errors are weighted by:
- **Volume**: high-selling items penalised more
- **Price**: high-revenue items penalised more
- **Scaled** by the naive 1-step forecast error (so persistent series are harder)

Lower is better. Top Kaggle scores are around **0.50**.

---

## Configuration (`config.py`)

| Setting | Default | Notes |
|---|---|---|
| `TIME_BUDGET_MINUTES` | `5` | Wall-clock limit per experiment |
| `MAX_EXPERIMENTS` | `200` | Stop after this many runs |
| `AGENT_MODEL` | `claude-sonnet-4-20250514` | Claude model for hypothesis generation |
| `AUTO_COMMIT` | `True` | Git commit on improvement |
| `VAL_START_DAY` | `1886` | Validation window start |

---

## Tuning the Research Directive

Edit `prompt.md` to steer the agent:

- **Narrow the hypothesis space**: "Focus only on feature engineering, do not change model params"
- **Force a direction**: "The next hypothesis must involve XGBoost instead of LightGBM"
- **Add domain knowledge**: "SNAP days in California show 40% higher variance — consider interaction terms"
- **Ban proven dead ends**: "Do NOT try learning_rate < 0.01 — this was tested in exp_0003 and timed out"

---

## File Structure

```
autoresearch_m5/
├── run.sh                   # Entry point
├── prompt.md                # Research directive (you edit this)
├── config.py                # Paths, budget, API settings
├── agent.py                 # Autonomous research loop
├── train.py                 # Current best training script (agent edits this)
├── train_best.py            # Copy of best train.py found so far
├── train_original.py        # Untouched baseline (auto-saved on first run)
├── data_prep.py             # M5 data loading & feature engineering utils
├── evaluate.py              # WRMSSE metric (stable)
├── data/                    # M5 CSV files
├── results/                 # preds_*.parquet + log_*.json per run
└── logs/
    ├── experiment_log.jsonl # Append-only experiment log
    └── train_*.py           # Proposed train.py backup per run
```

---

## Notes

- **Revert on failure**: if train.py crashes or times out, the harness automatically
  reverts to `train_best.py` so the agent always starts from a working baseline.
- **One hypothesis at a time**: the prompt explicitly instructs the agent to make
  a single isolated change — this ensures attribution of improvements.
- **Git history**: every improvement is committed, giving you a full research trail.
- **Memory**: the last 10 experiment entries are fed to the agent in every prompt,
  giving it context to avoid dead ends and build on prior gains.
