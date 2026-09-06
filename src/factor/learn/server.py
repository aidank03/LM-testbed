"""Local, dependency-light web server for the Factor Learning Lab."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import webbrowser

import torch

from .experiments import run_experiment
from .store import LearningRun


MAX_BODY_BYTES = 16 * 1024
STATIC_ROOT = Path(__file__).with_name("static")


class LearningLab:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.run_root = self.workspace / "runs" / "learning-lab"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, LearningRun] = {}
        self.lock = threading.Lock()

    def status(self):
        jax_spec = importlib.util.find_spec("jax")
        return {
            "python_engine": "local",
            "torch_version": torch.__version__,
            "device": (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            ),
            "jax": {
                "installed": jax_spec is not None,
                "executed_in_this_lab": False,
                "message": (
                    "Available for a future parity lesson."
                    if jax_spec is not None
                    else "Optional and not installed; the current lessons use PyTorch."
                ),
            },
            "workspace": str(self.workspace),
            "run_root": str(self.run_root),
            "lessons": ["line_fit", "motion_loop"],
        }

    def start(self, kind, config):
        if kind not in {"line_fit", "motion_loop"}:
            raise ValueError("kind must be line_fit or motion_loop")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        run = LearningRun(self.run_root, kind, config)
        with self.lock:
            self.runs[run.id] = run

        def worker():
            try:
                run.complete(run_experiment(kind, config, run.emit))
            except Exception as exc:  # preserve the failed experiment as evidence
                run.fail(exc)

        threading.Thread(target=worker, name=run.id, daemon=True).start()
        return run.snapshot()

    def list_runs(self):
        with self.lock:
            return [run.snapshot() for run in reversed(list(self.runs.values()))]

    def get_run(self, run_id):
        with self.lock:
            run = self.runs.get(run_id)
        return run.snapshot() if run else None


def _json(handler, status, value):
    body = json.dumps(value, allow_nan=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(lab: LearningLab):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                return _json(self, 200, lab.status())
            if path == "/api/runs":
                return _json(self, 200, {"runs": lab.list_runs()})
            if path.startswith("/api/runs/"):
                run_id = path.removeprefix("/api/runs/")
                if "/" in run_id or not run_id:
                    return _json(self, 404, {"error": "run not found"})
                run = lab.get_run(run_id)
                return _json(self, 200, run) if run else _json(self, 404, {"error": "run not found"})

            relative = "index.html" if path == "/" else path.lstrip("/")
            if relative not in {"index.html", "style.css", "app.js"}:
                return _json(self, 404, {"error": "not found"})
            file_path = STATIC_ROOT / relative
            if not file_path.is_file():
                return _json(self, 404, {"error": "asset not found"})
            content_types = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
            }
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_types[file_path.suffix])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/runs":
                return _json(self, 404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError("request body must be between 1 byte and 16 KiB")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {"kind", "config"}:
                    raise ValueError("request must contain exactly kind and config")
                return _json(self, 202, lab.start(payload["kind"], payload["config"]))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return _json(self, 400, {"error": str(exc)})

        def log_message(self, format, *args):
            return

    return Handler


def find_port(host, preferred):
    for port in range(preferred, preferred + 10):
        with socket.socket() as probe:
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free local port from {preferred} to {preferred + 9}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Factor PyTorch Learning Lab")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("the learning lab is intentionally limited to localhost")
    port = find_port("127.0.0.1", args.port)
    lab = LearningLab(args.workspace)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(lab))
    url = f"http://127.0.0.1:{port}"
    print(f"Factor Learning Lab: {url}", flush=True)
    print("Keep this terminal open while using the lab. Press Control-C to stop.", flush=True)
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFactor Learning Lab stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
