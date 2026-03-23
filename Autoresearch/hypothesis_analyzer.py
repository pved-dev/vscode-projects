"""
hypothesis_analyzer.py — Offline hypothesis improvement tool.

Reads experiment_log.jsonl, sends all hypothesis/result pairs to Claude,
and outputs a ranked list of better hypotheses to try next.

Run with: python3 hypothesis_analyzer.py
"""
import json
import os
from pathlib import Path
import anthropic

LOG_FILE = Path("logs/experiment_log.jsonl")

# ── Load experiment history ───────────────────────────────────────────────────
def load_all_experiments():
    if not LOG_FILE.exists():
        print("No experiment log found.")
        return []
    entries = []
    for line in LOG_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries

# ── Build analysis prompt ─────────────────────────────────────────────────────
def build_analysis_prompt(entries):
    valid   = [e for e in entries if e.get("wrmsse")]
    failed  = [e for e in entries if not e.get("wrmsse")]
    improved = [e for e in entries if e.get("improvement")]
    best    = min(valid, key=lambda e: e["wrmsse"]) if valid else None

    # Format all experiments
    exp_lines = []
    for e in entries:
        wrmsse = f"{e['wrmsse']:.5f}" if e.get("wrmsse") else "FAILED"
        flag   = "★ IMPROVED" if e.get("improvement") else ("✗ FAILED" if not e.get("wrmsse") else "no gain")
        exp_lines.append(f"[{e['run_id']}] WRMSSE={wrmsse} {flag}\n  Hypothesis: {e.get('hypothesis','')[:200]}")

    prompt = f"""You are an expert ML research strategist specializing in time-series forecasting with gradient boosted trees.

## Competition Context
- Kaggle M5 Forecasting: forecast 28 days of Walmart unit sales
- Metric: WRMSSE (lower is better)
- Models: LightGBM and XGBoost
- Current best WRMSSE: {best['wrmsse']:.5f} ({best['run_id']})
- Target: reach 0.90 or below

## Experiment History ({len(entries)} total, {len(improved)} improvements, {len(failed)} failures)

{chr(10).join(exp_lines)}

## Your Task
Analyze ALL experiments above. Identify:
1. Which directions produced improvements and why
2. Which directions are exhausted (tried many times, no gain)
3. Which high-value directions from top M5 solutions have NOT been explored

Then output exactly 10 concrete, specific hypothesis suggestions ranked by expected improvement.

For each hypothesis output:
- HYPOTHESIS: one clear sentence describing exactly what to change
- RATIONALE: why this should improve WRMSSE based on the experiment history
- IMPLEMENTATION: specific code pattern or approach
- EXPECTED_GAIN: estimated WRMSSE reduction (e.g. 0.005-0.010)
- PRIORITY: HIGH / MEDIUM / LOW

Format each as:
---
RANK: N
HYPOTHESIS: ...
RATIONALE: ...
IMPLEMENTATION: ...
EXPECTED_GAIN: ...
PRIORITY: ...
---

Be specific and concrete. Do not suggest anything already tried.
Focus on the highest-impact untried directions from top M5 competition solutions.
"""
    return prompt

# ── Call Claude ───────────────────────────────────────────────────────────────
def generate_hypotheses(prompt):
    try:
        import API
    except ImportError:
        pass

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    print("[analyzer] Calling Claude to analyze experiments …")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=(
            "You are an expert ML research strategist. "
            "Analyze experiment histories and suggest the highest-value next experiments. "
            "Be specific, concrete, and data-driven."
        ),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ── Save output ───────────────────────────────────────────────────────────────
def save_output(text, entries):
    output_path = Path("logs/hypothesis_suggestions.md")

    valid    = [e for e in entries if e.get("wrmsse")]
    improved = [e for e in entries if e.get("improvement")]
    best     = min(valid, key=lambda e: e["wrmsse"]) if valid else None

    header = f"""# Hypothesis Suggestions — AutoResearch M5
Generated from {len(entries)} experiments ({len(improved)} improvements)
Current best WRMSSE: {best['wrmsse']:.5f} ({best['run_id']})

---

{text}

---

## How to use these suggestions
1. Pick the hypotheses marked HIGH priority
2. Add them to the `## Next Priority` section in `prompt.md`
3. The agent will try them in the next run
4. After each run, re-run this tool to get updated suggestions

## Exhausted directions (do not retry)
"""
    # Auto-detect exhausted directions
    from collections import Counter
    hyp_words = []
    for e in entries:
        hyp = e.get("hypothesis", "")[:80]
        hyp_words.append(hyp)

    counts = Counter(hyp_words)
    exhausted = [h for h, c in counts.most_common(20) if c >= 3]
    for h in exhausted:
        header += f"- {h[:80]}\n"

    output_path.write_text(header)
    print(f"\n[analyzer] Suggestions saved to {output_path}")
    return output_path

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  AutoResearch M5 — Hypothesis Analyzer")
    print("="*60 + "\n")

    entries = load_all_experiments()
    if not entries:
        print("No experiments found in logs/experiment_log.jsonl")
        return

    valid    = [e for e in entries if e.get("wrmsse")]
    improved = [e for e in entries if e.get("improvement")]
    best     = min(valid, key=lambda e: e["wrmsse"]) if valid else None

    print(f"Loaded {len(entries)} experiments")
    print(f"  Successful : {len(valid)}")
    print(f"  Improved   : {len(improved)} ({len(improved)/len(valid)*100:.0f}% keep rate)")
    print(f"  Best WRMSSE: {best['wrmsse']:.5f} — {best['hypothesis'][:80]}" if best else "  No valid runs")
    print()

    prompt = build_analysis_prompt(entries)
    result = generate_hypotheses(prompt)

    if not result:
        return

    print("\n" + "="*60)
    print("  SUGGESTED HYPOTHESES")
    print("="*60)
    print(result)

    output_path = save_output(result, entries)
    print(f"\nFull output saved to: {output_path}")
    print("\nNext step: copy HIGH priority hypotheses into prompt.md → Next Priority section")

if __name__ == "__main__":
    main()