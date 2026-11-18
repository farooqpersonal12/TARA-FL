"""
TARA-FL Live Dashboard Backend (Mode A: local execution).

Serves the upgraded interactive dashboard (demo/index.html) and streams REAL
experiment metrics from the running Python pipeline to the browser via
Server-Sent Events (SSE). The metrics come exclusively from the JSON files
published by experiments/results_export.py during actual training runs
(dashboard_data/live.json and dashboard_data/history/*.json).

The backend NEVER generates metrics. If no experiment is running (or no
dashboard_data files exist), the API reports that state and the dashboard
shows it - no simulated data is substituted.

Endpoints
---------
GET /                          -> the dashboard (demo/index.html)
GET /results/<path>            -> published static results (demo/results/*)
GET /api/status                -> backend status + current experiment state
GET /api/live                  -> full live experiment document (live.json)
GET /api/stream                -> SSE stream: snapshot, then update per round
GET /api/experiments            -> experiment history index (index.json)
GET /api/experiments/<id>      -> one experiment's full document (history/<id>.json)

Usage
-----
    python visualization/live_server.py --port 8000
    (then open http://127.0.0.1:8000)

Run the experiment in a second terminal:
    python experiments/run_unified.py --method all --rounds 5 ...

Requires FastAPI + uvicorn (recommended, installed via requirements.txt).
Falls back to the Python standard library http.server if they are unavailable.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, AsyncIterator, Dict, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_DASHBOARD_DIR = os.environ.get("TARAFL_DASHBOARD_DIR", "dashboard_data")
DEFAULT_DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
DEFAULT_PORT = 8000

SSE_POLL_INTERVAL_SEC = 1.0
SSE_HEARTBEAT_SEC = 15.0

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False


# ======================================================================
# Core data access (shared by the FastAPI and stdlib implementations)
# ======================================================================

class DashboardStore:
    """Reads the exporter's JSON output. Read-only; never fabricates data."""

    def __init__(self, dashboard_dir: str, demo_dir: str):
        self.dashboard_dir = dashboard_dir
        self.demo_dir = demo_dir

    # ---------- paths ----------

    @property
    def live_path(self) -> str:
        return os.path.join(self.dashboard_dir, "live.json")

    @property
    def index_path(self) -> str:
        return os.path.join(self.dashboard_dir, "index.json")

    def history_path(self, experiment_id: str) -> str:
        # Prevent path traversal: ids are slugs produced by the exporter
        safe_id = os.path.basename(experiment_id).replace("\\", "").replace("/", "")
        return os.path.join(self.dashboard_dir, "history", f"{safe_id}.json")

    # ---------- file signature for change detection ----------

    @staticmethod
    def _signature(path: str) -> Optional[Tuple[int, int]]:
        try:
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None

    # ---------- documents ----------

    def _read_json(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None  # missing or (transiently) mid-replace -> treat as absent

    def live_doc(self) -> Optional[Dict[str, Any]]:
        return self._read_json(self.live_path)

    def index_doc(self) -> Optional[Dict[str, Any]]:
        return self._read_json(self.index_path)

    def experiment_doc(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self.history_path(experiment_id))

    def status_doc(self) -> Dict[str, Any]:
        live = self.live_doc()
        return {
            "backend": "online",
            "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dashboard_dir": os.path.abspath(self.dashboard_dir),
            "data_available": live is not None,
            "live": {
                "experiment_id": (live or {}).get("experiment_id"),
                "status": (live or {}).get("status"),
                "updated_at": (live or {}).get("updated_at"),
                "methods": sorted((live or {}).get("methods", {}).keys()),
                "rounds_completed": {
                    m: {
                        seed: len(run.get("rounds", []))
                        for seed, run in m_doc.get("runs", {}).items()
                    }
                    for m, m_doc in (live or {}).get("methods", {}).items()
                },
            } if live else None,
        }

    # ---------- static demo files ----------

    def demo_index_path(self) -> str:
        return os.path.join(self.demo_dir, "index.html")

    def resolve_demo_path(self, sub_path: str) -> Optional[str]:
        base = os.path.abspath(self.demo_dir)
        candidate = os.path.abspath(os.path.join(base, sub_path))
        if candidate.startswith(base) and os.path.isfile(candidate):
            return candidate
        return None


# ======================================================================
# SSE stream (shared logic)
# ======================================================================

def sse_event(event: str, payload: Any, event_id: Optional[str] = None) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    lines = [f"event: {event}"]
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {data}")
    return "\n".join(lines) + "\n\n"


# ======================================================================
# FastAPI application (primary backend)
# ======================================================================

def create_app(store: DashboardStore) -> "FastAPI":
    app = FastAPI(title="TARA-FL Live Dashboard Backend", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def api_status():
        return JSONResponse(store.status_doc())

    @app.get("/api/live")
    def api_live():
        doc = store.live_doc()
        if doc is None:
            raise HTTPException(status_code=404, detail={
                "error": "no_live_experiment",
                "message": ("No dashboard_data/live.json found. Start an experiment with "
                            "python experiments/run_unified.py ..."),
            })
        return JSONResponse(doc)

    @app.get("/api/experiments")
    def api_experiments():
        doc = store.index_doc()
        if doc is None:
            return JSONResponse({"updated_at": None, "count": 0, "experiments": []})
        return JSONResponse(doc)

    @app.get("/api/experiments/{experiment_id}")
    def api_experiment(experiment_id: str):
        doc = store.experiment_doc(experiment_id)
        if doc is None:
            raise HTTPException(status_code=404, detail={
                "error": "experiment_not_found",
                "message": f"No history document for experiment '{experiment_id}'.",
            })
        return JSONResponse(doc)

    @app.get("/api/stream")
    async def api_stream():
        async def event_stream() -> AsyncIterator[str]:
            last_sig: Optional[Tuple[int, int]] = None
            last_heartbeat = time.monotonic()
            first = True
            try:
                while True:
                    sig = store._signature(store.live_path)
                    if sig != last_sig:
                        last_sig = sig
                        doc = store.live_doc()
                        payload = {"live": doc, "server_time": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                        event = "snapshot" if first else "update"
                        if first:
                            first = False
                        exp_id = (doc or {}).get("experiment_id")
                        yield sse_event(event, payload, event_id=exp_id)
                        last_heartbeat = time.monotonic()
                    elif time.monotonic() - last_heartbeat >= SSE_HEARTBEAT_SEC:
                        yield "event: ping\ndata: {}\n\n"
                        last_heartbeat = time.monotonic()
                    await asyncio.sleep(SSE_POLL_INTERVAL_SEC)
            except asyncio.CancelledError:
                # client disconnected: uvicorn cancels the generator task
                return
            except Exception:
                return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/")
    def dashboard_index():
        index = store.demo_index_path()
        if not os.path.isfile(index):
            raise HTTPException(status_code=404, detail="demo/index.html not found")
        return FileResponse(index, media_type="text/html")

    @app.get("/results/{sub_path:path}")
    def published_results(sub_path: str):
        resolved = store.resolve_demo_path(os.path.join("results", sub_path))
        if resolved is None:
            raise HTTPException(status_code=404, detail={
                "error": "not_published",
                "message": ("No published results file. Run "
                            "python visualization/publish_results.py to export "
                            "dashboard_data into demo/results/."),
            })
        return FileResponse(resolved)

    return app


def run_fastapi(store: DashboardStore, host: str, port: int) -> None:
    app = create_app(store)
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ======================================================================
# Standard-library fallback backend (no external dependencies)
# ======================================================================

def make_stdlib_server(store: DashboardStore, host: str = "127.0.0.1", port: int = 0):
    """Create (not start) the stdlib fallback HTTPServer. port=0 picks a free port.
    Returned server exposes .server_address and must be shutdown() by the caller."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, payload: Any, code: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: str, content_type: str) -> None:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                if path == "/api/status":
                    self._send_json(store.status_doc())
                elif path == "/api/live":
                    doc = store.live_doc()
                    if doc is None:
                        self._send_json({"detail": {"error": "no_live_experiment"}}, 404)
                    else:
                        self._send_json(doc)
                elif path == "/api/experiments":
                    self._send_json(store.index_doc() or {
                        "updated_at": None, "count": 0, "experiments": []})
                elif path.startswith("/api/experiments/"):
                    exp_id = urllib.parse.unquote(path[len("/api/experiments/"):])
                    doc = store.experiment_doc(exp_id)
                    if doc is None:
                        self._send_json({"detail": {"error": "experiment_not_found"}}, 404)
                    else:
                        self._send_json(doc)
                elif path == "/api/stream":
                    self._serve_sse()
                elif path == "/":
                    index = store.demo_index_path()
                    if os.path.isfile(index):
                        self._send_file(index, "text/html; charset=utf-8")
                    else:
                        self._send_json({"detail": "demo/index.html not found"}, 404)
                elif path.startswith("/results/"):
                    resolved = store.resolve_demo_path(
                        urllib.parse.unquote(path[len("/results/"):]))
                    if resolved is None:
                        self._send_json({"detail": {"error": "not_published"}}, 404)
                    else:
                        ctype = "application/json" if resolved.endswith(".json") else "text/plain"
                        self._send_file(resolved, ctype)
                else:
                    self._send_json({"detail": "Not Found"}, 404)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _serve_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            last_sig = None
            first = True
            last_heartbeat = time.monotonic()
            try:
                while True:
                    sig = store._signature(store.live_path)
                    if sig != last_sig:
                        last_sig = sig
                        doc = store.live_doc()
                        payload = {"live": doc, "server_time": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                        event = "snapshot" if first else "update"
                        first = False
                        self.wfile.write(sse_event(event, payload,
                                                   event_id=(doc or {}).get("experiment_id")).encode("utf-8"))
                        last_heartbeat = time.monotonic()
                    elif time.monotonic() - last_heartbeat >= SSE_HEARTBEAT_SEC:
                        self.wfile.write(b"event: ping\ndata: {}\n\n")
                        last_heartbeat = time.monotonic()
                    self.wfile.flush()
                    time.sleep(SSE_POLL_INTERVAL_SEC)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # keep the console clean during long streams

    return ThreadingHTTPServer((host, port), Handler)


def run_stdlib(store: DashboardStore, host: str, port: int) -> None:
    server = make_stdlib_server(store, host, port)
    print(f"[*] TARA-FL dashboard (stdlib fallback) at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TARA-FL Live Dashboard Backend (SSE)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dashboard_dir", type=str, default=DEFAULT_DASHBOARD_DIR,
                        help="Directory containing live.json / history / index.json")
    parser.add_argument("--demo_dir", type=str, default=DEFAULT_DEMO_DIR,
                        help="Directory containing the demo/index.html dashboard")
    args = parser.parse_args()

    store = DashboardStore(dashboard_dir=args.dashboard_dir, demo_dir=args.demo_dir)

    print("=" * 65)
    print("        TARA-FL LIVE DASHBOARD BACKEND (real experiment feed)")
    print("=" * 65)
    print(f"  Dashboard URL : http://{args.host}:{args.port}/")
    print(f"  Data directory: {os.path.abspath(args.dashboard_dir)}")
    print(f"  Backend mode  : {'FastAPI + uvicorn' if FASTAPI_AVAILABLE else 'stdlib fallback (fastapi/uvicorn not installed)'}")
    print("  Start an experiment in another terminal:")
    print("    python experiments/run_unified.py --method all --rounds 5 ...")
    print("=" * 65)

    if FASTAPI_AVAILABLE:
        run_fastapi(store, args.host, args.port)
    else:
        run_stdlib(store, args.host, args.port)


if __name__ == "__main__":
    main()
