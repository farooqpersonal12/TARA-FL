"""
TARA-FL Interactive Live Web Dashboard.

Provides a self-contained, real-time web dashboard using Python's standard http.server.
Renders responsive dynamic charts (Chart.js + Tailwind CSS) for:
1. Accuracy & Loss Convergence across Baselines A-D
2. Per-Client Dynamic Trust Score Trajectories & Quarantine Status
3. Round-Risk Index & Adaptive Aggregator Strategy Switch Timeline
4. JSON REST API endpoints (/api/metrics, /api/trust, /api/status)
"""

import os
import csv
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, List, Any, Optional
import urllib.parse


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler providing REST API and Interactive Dashboard UI."""

    data_dir: str = "results"

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/" or path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = self.generate_dashboard_html()
            self.wfile.write(html.encode("utf-8"))
        elif path == "/api/status":
            self.send_json_response({"status": "running", "data_dir": self.data_dir})
        elif path == "/api/runs":
            runs = self.list_available_runs()
            self.send_json_response({"runs": runs})
        elif path == "/api/metrics":
            query = urllib.parse.parse_qs(parsed_path.query)
            run_file = query.get("file", ["tara_fl_results.csv"])[0]
            metrics = self.load_run_metrics(run_file)
            self.send_json_response(metrics)
        else:
            self.send_error(404, "Not Found")

    def send_json_response(self, data: Any):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def list_available_runs(self) -> List[str]:
        if not os.path.exists(self.data_dir):
            return []
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        return files

    def load_run_metrics(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            # Fallback mock/empty data
            return {"rounds": [], "accuracy": [], "loss": [], "risk": [], "trust": {}}

        rounds = []
        accuracies = []
        losses = []
        risks = []
        trust = {}

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            trust_cols = [col for col in fieldnames if col.startswith("trust_client_")]

            for col in trust_cols:
                cid = col.replace("trust_client_", "")
                trust[cid] = []

            for row in reader:
                r = int(row.get("round", 0))
                rounds.append(r)
                accuracies.append(float(row.get("accuracy_percent", float(row.get("accuracy", 0.0)) * 100)))
                losses.append(float(row.get("loss", 0.0)))
                risks.append(float(row.get("risk_score", 0.0)))

                for col in trust_cols:
                    cid = col.replace("trust_client_", "")
                    val = row.get(col, "0")
                    try:
                        trust[cid].append(float(val) if val != "N/A" else 0.0)
                    except ValueError:
                        trust[cid].append(0.0)

        return {
            "rounds": rounds,
            "accuracy": accuracies,
            "loss": losses,
            "risk": risks,
            "trust": trust
        }

    def generate_dashboard_html(self) -> str:
        runs = self.list_available_runs()
        runs_options = "".join([f'<option value="{r}">{r}</option>' for r in runs]) or '<option value="sample">No CSV files found in results/</option>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARA-FL Live Orchestration Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-900 text-slate-100 font-sans p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Top Nav -->
        <div class="bg-slate-800/80 backdrop-blur rounded-xl p-6 border border-slate-700 flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center gap-3">
                <div class="w-4 h-4 rounded-full bg-emerald-500 animate-pulse"></div>
                <h1 class="text-2xl font-bold tracking-tight">TARA-FL System Telemetry Dashboard</h1>
            </div>
            <div class="flex items-center gap-3">
                <label class="text-sm font-medium text-slate-400">Select Run:</label>
                <select id="runSelect" class="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-white focus:ring-2 focus:ring-indigo-500" onchange="loadRunData()">
                    {runs_options}
                </select>
                <button onclick="loadRunData()" class="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold rounded-lg shadow">Refresh</button>
            </div>
        </div>

        <!-- Telemetry Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="bg-slate-800 rounded-xl p-5 border border-slate-700">
                <p class="text-xs font-semibold uppercase tracking-wider text-slate-400">Current Accuracy</p>
                <p id="currAcc" class="text-3xl font-extrabold text-emerald-400 mt-2">--%</p>
            </div>
            <div class="bg-slate-800 rounded-xl p-5 border border-slate-700">
                <p class="text-xs font-semibold uppercase tracking-wider text-slate-400">Test Loss</p>
                <p id="currLoss" class="text-3xl font-extrabold text-indigo-400 mt-2">--</p>
            </div>
            <div class="bg-slate-800 rounded-xl p-5 border border-slate-700">
                <p class="text-xs font-semibold uppercase tracking-wider text-slate-400">Round Risk Score</p>
                <p id="currRisk" class="text-3xl font-extrabold text-amber-400 mt-2">--</p>
            </div>
            <div class="bg-slate-800 rounded-xl p-5 border border-slate-700">
                <p class="text-xs font-semibold uppercase tracking-wider text-slate-400">Completed Rounds</p>
                <p id="currRound" class="text-3xl font-extrabold text-sky-400 mt-2">--</p>
            </div>
        </div>

        <!-- Charts Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-slate-800 rounded-xl p-5 border border-slate-700">
                <h3 class="text-base font-bold text-slate-200 mb-4">Accuracy & Loss Convergence</h3>
                <div class="h-72"><canvas id="convergenceChart"></canvas></div>
            </div>
            <div class="bg-slate-800 rounded-xl p-5 border border-slate-700">
                <h3 class="text-base font-bold text-slate-200 mb-4">Environmental Round-Risk Timeline</h3>
                <div class="h-72"><canvas id="riskChart"></canvas></div>
            </div>
            <div class="bg-slate-800 rounded-xl p-5 border border-slate-700 lg:col-span-2">
                <h3 class="text-base font-bold text-slate-200 mb-4">Per-Client Dynamic Trust Score Trajectories</h3>
                <div class="h-80"><canvas id="trustChart"></canvas></div>
            </div>
        </div>
    </div>

    <script>
        let convChart, riskChart, trustChart;

        async function loadRunData() {{
            const select = document.getElementById('runSelect');
            const file = select ? select.value : '';
            if (!file) return;

            try {{
                const res = await fetch(`/api/metrics?file=${{encodeURIComponent(file)}}`);
                const data = await res.json();
                updateUI(data);
            }} catch(e) {{
                console.error("Failed to load metrics:", e);
            }}
        }}

        function updateUI(data) {{
            const rounds = data.rounds || [];
            if (rounds.length === 0) return;

            const lastIdx = rounds.length - 1;
            document.getElementById('currAcc').innerText = (data.accuracy[lastIdx] || 0).toFixed(2) + '%';
            document.getElementById('currLoss').innerText = (data.loss[lastIdx] || 0).toFixed(4);
            document.getElementById('currRisk').innerText = (data.risk[lastIdx] || 0).toFixed(3);
            document.getElementById('currRound').innerText = rounds[lastIdx];

            // Render Convergence Chart
            if (convChart) convChart.destroy();
            convChart = new Chart(document.getElementById('convergenceChart'), {{
                type: 'line',
                data: {{
                    labels: rounds,
                    datasets: [
                        {{ label: 'Accuracy (%)', data: data.accuracy, borderColor: '#10b981', tension: 0.2, yAxisID: 'y' }},
                        {{ label: 'Loss', data: data.loss, borderColor: '#6366f1', tension: 0.2, yAxisID: 'y1' }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{
                        y: {{ type: 'linear', display: true, position: 'left', min: 0, max: 100 }},
                        y1: {{ type: 'linear', display: true, position: 'right', grid: {{ drawOnChartArea: false }} }}
                    }}
                }}
            }});

            // Render Risk Chart
            if (riskChart) riskChart.destroy();
            riskChart = new Chart(document.getElementById('riskChart'), {{
                type: 'line',
                data: {{
                    labels: rounds,
                    datasets: [{{ label: 'Risk Score', data: data.risk, borderColor: '#f59e0b', backgroundColor: '#f59e0b20', fill: true, tension: 0.2 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ y: {{ min: 0, max: 1 }} }}
                }}
            }});

            // Render Trust Chart
            const trustDatasets = [];
            const palette = ['#10b981', '#3b82f6', '#ec4899', '#f97316', '#8b5cf6', '#ef4444', '#14b8a6', '#06b6d4', '#eab308', '#a855f7'];
            let cIdx = 0;
            for (const [cid, scores] of Object.entries(data.trust || {{}})) {{
                trustDatasets.push({{
                    label: `Client ${{cid}}`,
                    data: scores,
                    borderColor: palette[cIdx % palette.length],
                    tension: 0.2,
                    borderWidth: 1.8
                }});
                cIdx++;
            }}

            if (trustChart) trustChart.destroy();
            trustChart = new Chart(document.getElementById('trustChart'), {{
                type: 'line',
                data: {{ labels: rounds, datasets: trustDatasets }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ y: {{ min: 0, max: 1, title: {{ display: true, text: 'Trust Score Tk' }} }} }}
                }}
            }});
        }}

        window.onload = loadRunData;
    </script>
</body>
</html>
"""


def start_dashboard_server(
        data_dir: str = "results",
        port: int = 8080,
        run_in_thread: bool = True
) -> Optional[HTTPServer]:
    """
    Launches the TARA-FL telemetry dashboard server.
    """
    DashboardRequestHandler.data_dir = data_dir
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    print(f"[*] TARA-FL Dashboard live at: http://127.0.0.1:{port}/")

    if run_in_thread:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        return httpd
    else:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()
            print("\nDashboard server stopped.")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TARA-FL Live Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind dashboard")
    parser.add_argument("--data", type=str, default="results", help="Directory containing experiment CSVs")
    args = parser.parse_args()

    start_dashboard_server(data_dir=args.data, port=args.port, run_in_thread=False)
