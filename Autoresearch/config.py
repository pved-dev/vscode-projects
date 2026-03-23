"""
config.py — Central configuration for M5 AutoResearch
"""
import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── M5 Data Files ─────────────────────────────────────────────────────────────
SALES_FILE       = DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE    = DATA_DIR / "calendar.csv"
PRICES_FILE      = DATA_DIR / "sell_prices.csv"
SAMPLE_SUB_FILE  = DATA_DIR / "sample_submission.csv"

# ── Competition Settings ──────────────────────────────────────────────────────
HORIZON        = 28          # Forecast horizon (days)
N_TRAIN_DAYS   = 1913        # Total training days in dataset
VAL_START_DAY  = 1886        # Validation window: days 1886–1913
VAL_END_DAY    = 1913

# ── AutoResearch Loop ─────────────────────────────────────────────────────────
TIME_BUDGET_MINUTES = 5      # Max wall-clock time per experiment
MAX_EXPERIMENTS     = 13   # Stop after this many runs
AGENT_MODEL         = "claude-sonnet-4-6"
AGENT_MAX_TOKENS    = 16000

# ── Git ───────────────────────────────────────────────────────────────────────
AUTO_COMMIT         = True   # Commit train.py to git on improvement
MAIN_BRANCH         = "main"
EXPERIMENT_BRANCH_PREFIX = "exp"

# ── Baseline Metrics ──────────────────────────────────────────────────────────
# Update after establishing a baseline run
BASELINE_WRMSSE     = None   # Will be set after first run
BEST_WRMSSE         = None   # Tracks best so far across all runs

# ── Anthropic API ─────────────────────────────────────────────────────────────
# The API key is handled by the environment variable ANTHROPIC_API_KEY
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Experiment Log ────────────────────────────────────────────────────────────
EXPERIMENT_LOG_FILE = LOGS_DIR / "experiment_log.jsonl"
BEST_TRAIN_PY       = ROOT / "train_best.py"
