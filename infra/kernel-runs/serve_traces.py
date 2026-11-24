#!/usr/bin/env python3
"""Serve the kernel explorer page with the stage-1 traces and run folders, on localhost only.

    python3 infra/kernel-runs/serve_traces.py [--port 9001]      then open http://127.0.0.1:9001/

  /                 infra/workload-modeling/build/kernel-explorer.html (run `python -m workload_modeling.build` first)
  /traces/<...>     $SHVLA_DATA/stage1-h1/traces/ (Perfetto JSON; the page posts them to ui.perfetto.dev)
  /runs/<...>       $SHVLA_DATA/stage1-h1/ run folders (run.log, kernel_cfg.py, ... as text)
  /api/ping         lets the page know the server is there

Read-only, bound to 127.0.0.1. Port 9001 is also the one ui.perfetto.dev accepts for ?url= links
(https://ui.perfetto.dev/#!/?url=http://127.0.0.1:9001/traces/...). From WSL2, Windows browsers reach it through
localhost forwarding.
"""
import argparse
import http.server
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "workload-modeling/build/kernel-explorer.html"
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1"
ROOTS = {"/traces/": DATA / "traces", "/runs/": DATA}
TEXT = {".log", ".py", ".h", ".txt", ".json", ".diff", ".c"}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html", "/kernel-explorer.html"):
            if not PAGE.exists():
                return self._send(404, b"build the page first: cd infra/workload-modeling && uv run python -m workload_modeling.build")
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/ping":
            return self._send(200, b"ok")
        for prefix, root in ROOTS.items():
            if path.startswith(prefix):
                f = (root / path[len(prefix):]).resolve()
                if not f.is_relative_to(root.resolve()) or not f.is_file():
                    return self._send(404, b"not found")
                ctype = "application/json" if f.suffix == ".json" else \
                    "text/plain; charset=utf-8" if f.suffix in TEXT else "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype)
        return self._send(404, b"not found")

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9001)
    a = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"kernel explorer: http://127.0.0.1:{a.port}/   (traces from {DATA / 'traces'}; Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
