#!/usr/bin/env python3
"""Instant-bind placeholder (stdlib ONLY — runs before any venv).

Binds the app port within a second of the container existing and
serves a live "starting" page (boot journal tails, auto-refresh),
so the Codespaces proxy ALWAYS has something answering: a user
sees progress or a traceback, never HTTP 502. The real server
replaces this process during auto_start's handover.

/        -> 200 status page      /healthz -> 200 {"stage": ...}
/api/*   -> 503 {"ok": false, "stage": "starting"}  (the real
app's health checks must not mistake this for the app)
"""
import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tail(rel, n=28):
    try:
        with open(os.path.join(ROOT, rel), errors="replace") as f:
            lines = f.readlines()[-n:]
        return "".join(lines).strip() or "(empty)"
    except OSError:
        return "(no log yet)"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/healthz"):
            return self._send(200, json.dumps(
                {"ok": True, "stage": "starting"}), "application/json")
        if self.path.startswith("/api/"):
            return self._send(503, json.dumps(
                {"ok": False, "stage": "starting",
                 "reason": "engine still starting — page auto-refreshes"}),
                "application/json")
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Starting…</title><meta http-equiv="refresh" content="3">
<style>body{{background:#0b0e13;color:#dbe4f0;font:14px system-ui;
margin:40px auto;max-width:820px;padding:0 18px}}
h1{{font-size:20px}} .ok{{color:#3fb96f}} pre{{background:#11161f;
border:1px solid #22304a;border-radius:8px;padding:12px;overflow:auto;
font-size:12px;white-space:pre-wrap}}</style></head><body>
<h1>⏳ Prediction app is starting…</h1>
<p>The engine is provisioning/booting in the background. This page
refreshes every 3 s and is replaced automatically by the real app.
You should never see a 502 — if this page persists for minutes,
the logs below say why.</p>
<h3>boot journal (auto_start)</h3><pre>{html.escape(_tail("data/auto_start.log"))}</pre>
<h3>provisioning (setup)</h3><pre>{html.escape(_tail("data/setup.log"))}</pre>
<h3>engine (server)</h3><pre>{html.escape(_tail("data/server.log"))}</pre>
</body></html>"""
        return self._send(200, page, "text/html; charset=utf-8")


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
