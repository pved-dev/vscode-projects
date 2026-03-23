#!/usr/bin/env bash
# run.sh — AutoResearch M5 entry point
# Usage:
#   ./run.sh setup   — install dependencies and initialise git
#   ./run.sh data    — download M5 data via Kaggle API
#   ./run.sh once    — run a single experiment manually (test mode)
#   ./run.sh loop    — start the full autonomous research loop
#   ./run.sh report  — print experiment log summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[run.sh]${NC} $*"; }
success() { echo -e "${GREEN}[run.sh]${NC} $*"; }
warn()    { echo -e "${YELLOW}[run.sh]${NC} $*"; }
error()   { echo -e "${RED}[run.sh]${NC} $*"; exit 1; }

# ── Check API key ───────────────────────────────────────────────────────────
check_api_key() {
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        error "ANTHROPIC_API_KEY is not set. Export it before running:\n  export ANTHROPIC_API_KEY=sk-ant-..."
    fi
    success "ANTHROPIC_API_KEY found."
}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
cmd_setup() {
    info "Installing Python dependencies …"
    pip install \
        anthropic \
        lightgbm \
        xgboost \
        pandas \
        numpy \
        pyarrow \
        scikit-learn \
        kaggle \
        tqdm \
        --quiet

    success "Dependencies installed."

    # Initialise git if needed
    if [[ ! -d .git ]]; then
        info "Initialising git repo …"
        git init
        git add .
        git commit -m "Initial commit — baseline autoresearch_m5"
        success "Git repo initialised."
    else
        info "Git repo already exists."
    fi

    # Create dirs
    mkdir -p data results logs
    success "Setup complete."
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
cmd_data() {
    info "Downloading M5 data via Kaggle API …"
    info "Ensure ~/.kaggle/kaggle.json is configured."

    if ! command -v kaggle &>/dev/null; then
        error "kaggle CLI not found. Run './run.sh setup' first."
    fi

    kaggle competitions download -c m5-forecasting-accuracy -p data/
    cd data && unzip -o m5-forecasting-accuracy.zip && cd ..

    success "M5 data downloaded to ./data/"
    ls -lh data/*.csv
}

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RUN (smoke test)
# ─────────────────────────────────────────────────────────────────────────────
cmd_once() {
    info "Running a single experiment with current train.py …"
    export RUN_ID="manual_$(date +%Y%m%d_%H%M%S)"
    venv/bin/python3 train.py
    success "Single run complete. Check results/log_${RUN_ID}.json"
}

# ─────────────────────────────────────────────────────────────────────────────
# FULL AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────
cmd_loop() {
    check_api_key
    info "Starting AutoResearch loop …"
    info "Press Ctrl-C to stop gracefully."
    venv/bin/python3 agent.py
}

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
cmd_report() {
    info "Experiment log summary:"
    venv/bin/python3 - <<'PYEOF'
import json
from pathlib import Path

log_path = Path("logs/experiment_log.jsonl")
if not log_path.exists():
    print("No experiments yet.")
    exit()

entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
if not entries:
    print("No experiments yet.")
    exit()

best = min((e for e in entries if e.get("wrmsse")), key=lambda e: e["wrmsse"], default=None)
print(f"\n{'─'*80}")
print(f"{'RUN ID':<40} {'WRMSSE':>8}  {'DUR':>6}  {'FLAG'}")
print(f"{'─'*80}")
for e in entries[-30:]:
    wrmsse_str = f"{e['wrmsse']:.5f}" if e.get('wrmsse') else "  FAIL "
    flag = "★ BEST" if best and e.get('wrmsse') == best['wrmsse'] else ("▲ NEW" if e.get('improvement') else "")
    print(f"{e['run_id']:<40} {wrmsse_str:>8}  {e['duration_sec']:>5.0f}s  {flag}")

print(f"{'─'*80}")
if best:
    print(f"\n★  Best run : {best['run_id']}")
    print(f"   WRMSSE   : {best['wrmsse']:.5f}")
    print(f"   Hypothesis: {best['hypothesis']}")
print(f"\nTotal experiments: {len(entries)}")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
CMD="${1:-help}"
case "$CMD" in
    setup)  cmd_setup  ;;
    data)   cmd_data   ;;
    once)   cmd_once   ;;
    loop)   cmd_loop   ;;
    report) cmd_report ;;
    *)
        echo -e "${BOLD}AutoResearch M5 — Usage:${NC}"
        echo "  ./run.sh setup   — Install dependencies, init git"
        echo "  ./run.sh data    — Download M5 data (needs Kaggle API key)"
        echo "  ./run.sh once    — Run a single experiment manually"
        echo "  ./run.sh loop    — Start the autonomous agent loop"
        echo "  ./run.sh report  — Print experiment log summary"
        ;;
esac
