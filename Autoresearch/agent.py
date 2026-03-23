import pandas as pd
import ast
"""
agent.py — AutoResearch loop for M5 Forecasting.

WRMSSE is computed independently here after each run.
The agent cannot game the metric by modifying train.py.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

from config import (
    AGENT_MODEL, AGENT_MAX_TOKENS, ANTHROPIC_API_KEY,
    AUTO_COMMIT, BEST_TRAIN_PY, EXPERIMENT_LOG_FILE,
    MAX_EXPERIMENTS, ROOT, TIME_BUDGET_MINUTES,
    RESULTS_DIR, VAL_START_DAY, VAL_END_DAY,
)
from data_prep import load_sales, load_calendar, load_prices, compute_wrmsse_weights
from evaluate import wrmsse, load_preds

# ── Pre-load actuals once at startup ─────────────────────────────────────────
print("[agent] Loading M5 data for independent evaluation …")
_sales    = load_sales()
_calendar = load_calendar()
_prices   = load_prices()
_weights, _scale = compute_wrmsse_weights(_sales, _prices, _calendar)
_day_cols = [f"d_{i}" for i in range(VAL_START_DAY, VAL_END_DAY + 1)]
_actuals  = _sales[["id"] + _day_cols].copy()
print("[agent] Ready.\n")

PROMPT_TEMPLATE_PATH = ROOT / "prompt.md"
TRAIN_PY_PATH        = ROOT / "train.py"


def read_file(path):
    return path.read_text(encoding="utf-8")

def write_file(path, content):
    path.write_text(content, encoding="utf-8")

def load_experiment_log(n_recent=20):
    if not EXPERIMENT_LOG_FILE.exists():
        return []
    entries = []
    with open(EXPERIMENT_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-n_recent:]

def append_to_log(entry):
    with open(EXPERIMENT_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def format_log_for_prompt(entries):
    if not entries:
        return "No experiments yet. This is the first run."

    # Deduplicated already-tried summary
    tried = []
    seen = set()
    for e in reversed(entries):
        key = e.get("hypothesis", "")[:60].strip()
        if key and key not in seen:
            seen.add(key)
            tried.append(key)

    summary = "ALREADY TRIED — DO NOT REPEAT THESE:\n"
    for t in tried[:20]:
        summary += f"  - {t}\n"
    summary += "\n"

    # Full log entries
    lines = []
    for e in reversed(entries):
        flag = "✓ IMPROVEMENT" if e.get("improvement") else ""
        wrmsse_str = f"{e['wrmsse']:.5f}" if e.get("wrmsse") is not None else "FAILED"
        lines.append(
            f"[{e['run_id']}] WRMSSE={wrmsse_str}  {e['duration_sec']:.0f}s  "
            f"{e['timestamp']}  {flag}\n"
            f"  Hypothesis: {e.get('hypothesis', '')[:120]}"
        )

    return summary + "\n".join(lines)

def build_prompt(log_entries):
    template = read_file(PROMPT_TEMPLATE_PATH)
    prompt = template.replace("{TIME_BUDGET_MINUTES}", str(TIME_BUDGET_MINUTES))
    prompt = prompt.replace("{EXPERIMENT_LOG}", format_log_for_prompt(log_entries))
    prompt = prompt.replace("{CURRENT_TRAIN_PY}", read_file(TRAIN_PY_PATH))
    return prompt

def call_agent(prompt, retries=3, wait=10):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = (
    "You are a Python code generator. "
    "Your response must start with 'import' on the very first line. "
    "Output ONLY raw Python code — no explanations, no markdown, no preamble. "
    "First character of your response must be 'i' from 'import json'. "
    "CRITICAL: Do NOT compute WRMSSE. Save predictions to results/preds_{RUN_ID}.parquet and print DONE."
)
    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=AGENT_MODEL, max_tokens=AGENT_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            code = response.content[0].text
            code = re.sub(r"^```(?:python)?\s*\n", "", code, flags=re.MULTILINE)
            code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)
            return code.strip()
        except Exception as e:
            print(f"[agent] API error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                print(f"[agent] Retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise

def run_experiment(run_id):
    env = {**os.environ, "RUN_ID": run_id}
    print(f"\n{'='*60}\n[agent] Running {run_id} …\n{'='*60}")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(TRAIN_PY_PATH)],
            capture_output=True, text=True,
            timeout=TIME_BUDGET_MINUTES * 60,
            env=env, cwd=str(ROOT),
        )
        duration = time.time() - t0
        if result.returncode != 0:
            print(f"[agent] ✗ Failed (exit {result.returncode})\n{result.stderr[-1000:]}")
            return False, duration
        if "DONE" not in result.stdout:
            print(f"[agent] ✗ train.py did not print DONE\n{result.stdout[-500:]}")
            return False, duration
        print(result.stdout)
        return True, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        print(f"[agent] ⏱ Timeout after {duration:.0f}s")
        return False, duration

def evaluate_run(run_id):
    """Compute WRMSSE independently from saved predictions."""
    pred_path = RESULTS_DIR / f"preds_{run_id}.parquet"
    if not pred_path.exists():
        print(f"[agent] ✗ No predictions file found")
        return None
    try:
        preds = load_preds(str(pred_path))
        score = wrmsse(preds, _actuals, _weights, _scale)
        if score < 0.3 or score > 10.0:
            print(f"[agent] ✗ Implausible WRMSSE={score:.5f} — rejecting")
            return None
        return score
    except Exception as e:
        print(f"[agent] ✗ Evaluation error: {e}")
        return None

def extract_hypothesis(code):
    match = re.search(r'HYPOTHESIS\s*=\s*["\'](.+?)["\']', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'HYPOTHESIS\s*=\s*\((.+?)\)', code, re.DOTALL)
    if match:
        return re.sub(r'\s+', ' ', match.group(1).replace('"','').replace("'",''))
    return "Unknown hypothesis"

def git_commit(run_id, score):
    if not AUTO_COMMIT:
        return
    try:
        subprocess.run(["git","add","train.py"], cwd=str(ROOT), check=True, capture_output=True)
        subprocess.run(["git","commit","-m",f"[autoresearch] {run_id} WRMSSE={score:.5f}"],
                       cwd=str(ROOT), check=True, capture_output=True)
        print(f"[agent] ✓ Committed train.py")
    except subprocess.CalledProcessError as e:
        print(f"[agent] Git commit failed: {e}")


def main():
    if not ANTHROPIC_API_KEY:
        print("[agent] ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print("█"*60)
    print("  M5 AutoResearch Loop")
    print(f"  Model: {AGENT_MODEL}")
    print(f"  Budget: {TIME_BUDGET_MINUTES} min/experiment")
    print(f"  Max experiments: {MAX_EXPERIMENTS}")
    print("█"*60 + "\n")

    log_entries = load_experiment_log()
    best_wrmsse = min((e["wrmsse"] for e in log_entries if e.get("wrmsse")), default=float("inf"))
    run_counter = len(log_entries)

    if run_counter == 0:
        shutil.copy(TRAIN_PY_PATH, ROOT / "train_original.py")
        print("[agent] Backed up train.py → train_original.py")

    for iteration in range(1, MAX_EXPERIMENTS + 1):
        run_id = f"exp_{run_counter:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n[agent] ─── Iteration {iteration}  ({run_id}) ───")

        # Step 1: Get new hypothesis from Claude
        print("[agent] Calling Claude …")
        new_code   = call_agent(build_prompt(load_experiment_log(n_recent=20)))
        hypothesis = extract_hypothesis(new_code)
        print(f"[agent] Hypothesis: {hypothesis[:100]}")

      # Step 2: Validate syntax before writing
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            print(f"[agent] ✗ Syntax error in generated code: {e}")
            print(f"[agent] Skipping this iteration — reverting to best model")
            entry = {"run_id": run_id, "hypothesis": hypothesis,
                     "wrmsse": None, "duration_sec": 0,
                     "timestamp": datetime.now().isoformat(),
                     "status": "syntax_error", "improvement": False}
            append_to_log(entry)
            run_counter += 1
            continue

        # Step 3: Back up and write new train.py
        backup = ROOT / "logs" / f"train_{run_id}_proposed.py"
        shutil.copy(TRAIN_PY_PATH, backup)
        write_file(TRAIN_PY_PATH, new_code)

        # Step 3: Run experiment
        success, duration = run_experiment(run_id)

        # Step 4: Independently evaluate
        score = evaluate_run(run_id) if success else None

        # Step 5: Keep or revert
        if score is not None:
            improvement = score < best_wrmsse
            delta = best_wrmsse - score if improvement else score - best_wrmsse
            symbol = "✓ IMPROVEMENT" if improvement else "✗ no gain"
            print(f"\n[agent] {symbol}  WRMSSE={score:.5f}  best={best_wrmsse:.5f}  Δ={delta:+.5f}")

            if improvement:
                best_wrmsse = score
                shutil.copy(TRAIN_PY_PATH, BEST_TRAIN_PY)
                git_commit(run_id, score)
                # Save best predictions as readable CSV
                pred_path = RESULTS_DIR / f"preds_{run_id}.parquet"
                preds_df = pd.read_parquet(pred_path)
                preds_df.to_csv(ROOT / "best_predictions.csv", index=False)
                print(f"[agent] ★ New best saved to train_best.py")
                print(f"[agent] ★ Predictions saved to best_predictions.csv")
            else:
                src = BEST_TRAIN_PY if BEST_TRAIN_PY.exists() else backup
                shutil.copy(src, TRAIN_PY_PATH)
                print("[agent] Reverted to best model")

            entry = {"run_id": run_id, "hypothesis": hypothesis,
                     "wrmsse": round(score, 6), "duration_sec": round(duration, 1),
                     "timestamp": datetime.now().isoformat(),
                     "status": "success", "improvement": improvement}
        else:
            src = BEST_TRAIN_PY if BEST_TRAIN_PY.exists() else backup
            shutil.copy(src, TRAIN_PY_PATH)
            print("[agent] ✗ Failed — reverted to best model")
            entry = {"run_id": run_id, "hypothesis": hypothesis,
                     "wrmsse": None, "duration_sec": round(duration, 1),
                     "timestamp": datetime.now().isoformat(),
                     "status": "failed", "improvement": False}

        append_to_log(entry)
        run_counter += 1

        # Print recent history
        recent = load_experiment_log(n_recent=5)
        print(f"\n[agent] Recent history:")
        for e in recent:
            flag = "★" if e.get("improvement") else " "
            ws = f"{e['wrmsse']:.5f}" if e.get("wrmsse") else "  FAIL "
            print(f"  {flag} {e['run_id']:<38} WRMSSE={ws}")
        print(f"[agent] Best: {best_wrmsse:.5f}\n")

    print(f"\n[agent] Done. Best WRMSSE: {best_wrmsse:.5f}")


if __name__ == "__main__":
    main()