"""
dashboard.py — Live AutoResearch M5 Dashboard
Serves a real-time web dashboard at http://localhost:5000

Run with: python3 dashboard.py
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG_FILE = Path("logs/experiment_log.jsonl")
PORT = 5000

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoResearch M5 — Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #111118;
    --border: #1e1e2e;
    --accent: #00ff87;
    --accent2: #ff6b6b;
    --accent3: #ffd93d;
    --text: #e0e0f0;
    --muted: #5a5a7a;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Mono', monospace;
    min-height: 100vh;
    padding: 2rem;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at 20% 20%, rgba(0,255,135,0.04) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 80%, rgba(255,107,107,0.04) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  .container { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; }

  header {
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
  }

  h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--accent);
  }

  .subtitle { color: var(--muted); font-size: 0.75rem; }

  .pulse {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
    margin-left: auto;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(0,255,135,0.4); }
    50% { opacity: 0.8; transform: scale(1.2); box-shadow: 0 0 0 6px rgba(0,255,135,0); }
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
  }

  .stat-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
  }

  .stat-card.red::after { background: var(--accent2); }
  .stat-card.yellow::after { background: var(--accent3); }
  .stat-card.blue::after { background: #60a5fa; }

  .stat-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
  .stat-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1; }
  .stat-value.green { color: var(--accent); }
  .stat-value.red { color: var(--accent2); }
  .stat-value.yellow { color: var(--accent3); }
  .stat-value.blue { color: #60a5fa; }
  .stat-sub { font-size: 0.65rem; color: var(--muted); margin-top: 0.4rem; }

  .chart-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    height: 280px;
  }

  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 1rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.75rem;
  }

  th {
    text-align: left;
    padding: 0.75rem 1rem;
    color: var(--muted);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }

  td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }

  tr:last-child td { border-bottom: none; }

  tr.improved td { background: rgba(0,255,135,0.04); }
  tr.failed td { background: rgba(255,107,107,0.04); opacity: 0.6; }

  .badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .badge.improved { background: rgba(0,255,135,0.15); color: var(--accent); }
  .badge.failed   { background: rgba(255,107,107,0.15); color: var(--accent2); }
  .badge.normal   { background: rgba(90,90,122,0.2); color: var(--muted); }
  .badge.best     { background: rgba(255,217,61,0.15); color: var(--accent3); }

  .wrmsse-good { color: var(--accent); }
  .wrmsse-bad  { color: var(--accent2); }
  .wrmsse-mid  { color: var(--accent3); }

  .hypothesis { color: var(--muted); max-width: 400px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .refresh-note { text-align: center; color: var(--muted); font-size: 0.65rem; margin-top: 1.5rem; }

  canvas { max-height: 220px !important; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>AutoResearch M5</h1>
    <span class="subtitle">LightGBM / XGBoost Autonomous Loop</span>
    <span class="pulse"></span>
  </header>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total Runs</div>
      <div class="stat-value blue" id="total">—</div>
      <div class="stat-sub" id="success-rate">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Best WRMSSE</div>
      <div class="stat-value green" id="best">—</div>
      <div class="stat-sub" id="best-run">—</div>
    </div>
    <div class="stat-card red">
      <div class="stat-label">Baseline WRMSSE</div>
      <div class="stat-value yellow" id="baseline">0.98144</div>
      <div class="stat-sub">manual baseline</div>
    </div>
    <div class="stat-card yellow">
      <div class="stat-label">Improvements</div>
      <div class="stat-value yellow" id="improvements">—</div>
      <div class="stat-sub" id="improve-pct">—</div>
    </div>
  </div>

  <div class="chart-container">
    <div class="section-title">WRMSSE over experiments</div>
    <canvas id="chart"></canvas>
  </div>

  <div class="section-title">Experiment Log</div>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Run ID</th>
        <th>WRMSSE</th>
        <th>Duration</th>
        <th>Status</th>
        <th>Hypothesis</th>
      </tr>
    </thead>
    <tbody id="log-body"></tbody>
  </table>

  <div class="refresh-note">Auto-refreshes every 10 seconds</div>
</div>

<script>
let chart = null;

function wrmssColor(v) {
  if (v === null) return '';
  if (v < 0.75) return 'wrmsse-good';
  if (v < 0.95) return 'wrmsse-mid';
  return 'wrmsse-bad';
}

async function refresh() {
  try {
    const res = await fetch('/data');
    const entries = await res.json();
    if (!entries.length) return;

    const valid = entries.filter(e => e.wrmsse !== null);
    const failed = entries.filter(e => e.wrmsse === null);
    const improved = entries.filter(e => e.improvement);
    const best = valid.length ? valid.reduce((a,b) => a.wrmsse < b.wrmsse ? a : b) : null;

    document.getElementById('total').textContent = entries.length;
    document.getElementById('success-rate').textContent = `${valid.length} success / ${failed.length} failed`;
    document.getElementById('best').textContent = best ? best.wrmsse.toFixed(5) : '—';
    document.getElementById('best-run').textContent = best ? best.run_id.slice(0,20) : '—';
    document.getElementById('improvements').textContent = improved.length;
    document.getElementById('improve-pct').textContent = valid.length ? `${((improved.length/valid.length)*100).toFixed(0)}% hit rate` : '—';

    // Chart
    const labels = valid.map((e,i) => i+1);
    const scores = valid.map(e => e.wrmsse);
    const colors = valid.map(e => e.improvement ? '#00ff87' : '#5a5a7a');

    const baseline = Array(valid.length).fill(0.98144);

    if (!chart) {
      const ctx = document.getElementById('chart').getContext('2d');
      chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'WRMSSE',
              data: scores,
              borderColor: '#60a5fa',
              backgroundColor: 'rgba(96,165,250,0.08)',
              pointBackgroundColor: colors,
              pointRadius: 5,
              pointHoverRadius: 7,
              tension: 0.3,
              fill: true,
            },
            {
              label: 'Baseline',
              data: baseline,
              borderColor: '#ffd93d',
              borderDash: [6,3],
              pointRadius: 0,
              tension: 0,
              fill: false,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#5a5a7a', font: { family: 'Space Mono', size: 11 } } },
            tooltip: { backgroundColor: '#111118', borderColor: '#1e1e2e', borderWidth: 1 }
          },
          scales: {
            x: { ticks: { color: '#5a5a7a', font: { family: 'Space Mono', size: 10 } }, grid: { color: '#1e1e2e' } },
            y: { ticks: { color: '#5a5a7a', font: { family: 'Space Mono', size: 10 } }, grid: { color: '#1e1e2e' }, reverse: false }
          }
        }
      });
    } else {
      chart.data.labels = labels;
      chart.data.datasets[0].data = scores;
      chart.data.datasets[0].pointBackgroundColor = colors;
      chart.data.datasets[1].data = baseline;
      chart.update();
    }

    // Table — show most recent first
    const tbody = document.getElementById('log-body');
    tbody.innerHTML = [...entries].reverse().map((e, i) => {
      const idx = entries.length - i;
      const wrmsse = e.wrmsse !== null ? `<span class="${wrmssColor(e.wrmsse)}">${e.wrmsse.toFixed(5)}</span>` : '<span class="wrmsse-bad">FAILED</span>';
      let badge = `<span class="badge normal">no gain</span>`;
      if (e.wrmsse === null) badge = `<span class="badge failed">failed</span>`;
      else if (best && e.wrmsse === best.wrmsse) badge = `<span class="badge best">★ best</span>`;
      else if (e.improvement) badge = `<span class="badge improved">↓ improved</span>`;
      const rowClass = e.improvement ? 'improved' : (e.wrmsse === null ? 'failed' : '');
      const hyp = (e.hypothesis || '').slice(0, 80);
      const dur = e.duration_sec ? `${e.duration_sec.toFixed(0)}s` : '—';
      return `<tr class="${rowClass}">
        <td>${idx}</td>
        <td>${e.run_id.slice(0,28)}</td>
        <td>${wrmsse}</td>
        <td>${dur}</td>
        <td>${badge}</td>
        <td class="hypothesis" title="${e.hypothesis}">${hyp}</td>
      </tr>`;
    }).join('');

  } catch(err) {
    console.error('Refresh error:', err);
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif self.path == "/data":
            entries = []
            if LOG_FILE.exists():
                for line in LOG_FILE.read_text().splitlines():
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(entries).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence access logs


if __name__ == "__main__":
    print(f"[dashboard] Starting at http://localhost:{PORT}")
    print(f"[dashboard] Reading from {LOG_FILE}")
    print(f"[dashboard] Press Ctrl-C to stop")
    server = HTTPServer(("localhost", PORT), Handler)
    server.serve_forever()