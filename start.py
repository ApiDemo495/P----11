#!/usr/bin/env python3
"""start.py — the ENTIRE boot system in one stdlib-only Python file.

Replaces the old shell chain (devcontainer hooks -> auto_start.sh
-> setup.sh -> supervise.sh -> placeholder + flock files + curl).
That chain kept wedging in real Codespaces: curl-based health
checks, a flock fd leaking into hung editor-CLI children, and
unbounded git/pip steps. This file depends on NOTHING but
python3 itself:

    python3 start.py            foreground: provision + serve + watch
    python3 start.py --detach   (devcontainer hooks) returns in <1 s;
                                a detached copy does the work
    python3 start.py --stop     stop the detached instance
    python3 start.py --status   print what is answering on the port

Flow inside --serve:
  1. a placeholder page binds the port in a thread and answers
     200 / with live boot logs (the proxy never finds a dead port)
  2. venv + core requirements are provisioned (stamp-cached)
  3. `import app.main` must pass BEFORE the placeholder is stopped
  4. uvicorn runs as a child; when /api/health answers, the
     placeholder thread shuts down and the app owns the port
  5. a watchdog restarts the child on crashes (placeholder
     returns during restarts); a hung child is killed after
     120 s of failed health checks
Logs: data/boot.log (this file), data/server.log (uvicorn).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
PORT = int(os.environ.get("PORT", "8000"))
DATA = os.path.join(ROOT, "data")
PIDFILE = os.path.join(DATA, "start.pid")
STAMP = os.path.join(ROOT, ".setup_stamp")
BOOTLOG = os.path.join(DATA, "boot.log")
SERVERLOG = os.path.join(DATA, "server.log")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
HEALTH = f"http://127.0.0.1:{PORT}/api/health"

_state = {"stage": "starting", "detail": "booting"}
_child: subprocess.Popen | None = None


def log(msg: str) -> None:
    line = f"[start {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


# ── tiny HTTP helpers (urllib only — no curl anywhere) ──────────────
def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        return 200 <= e.code < 300
    except Exception:
        return False


def healthy() -> bool:
    return http_ok(HEALTH, 3.0)


# ── placeholder: answers from second ~0 until the app is ready ─────
def _tail(path: str, n: int = 26) -> str:
    try:
        with open(path, errors="replace") as fh:
            return "".join(fh.readlines()[-n:]).strip() or "(empty)"
    except OSError:
        return "(no log yet)"


class _Placeholder(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/healthz"):
            return self._send(200, json.dumps(
                {"ok": True, "stage": _state["stage"]}))
        if self.path.startswith("/api/"):
            return self._send(503, json.dumps(
                {"ok": False, "stage": _state["stage"],
                 "reason": "engine still starting — page auto-refreshes"}))
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Starting…</title><meta http-equiv="refresh" content="3">
<style>body{{background:#0b0e13;color:#dbe4f0;font:14px system-ui;
margin:40px auto;max-width:820px;padding:0 18px}}
pre{{background:#11161f;border:1px solid #22304a;border-radius:8px;
padding:12px;overflow:auto;font-size:12px;white-space:pre-wrap}}</style>
</head><body><h1>⏳ app is starting — {_state['stage']}</h1>
<p>{_state['detail']}. This page refreshes itself; the app takes
over the moment it is healthy. Downloads, if any, happen here —
watch the logs below, not the terminal.</p>
<h3>boot log</h3><pre>{_tail(BOOTLOG)}</pre>
<h3>server log</h3><pre>{_tail(SERVERLOG)}</pre>
</body></html>"""
        return self._send(200, page, "text/html; charset=utf-8")


class _PlaceholderServer:
    def __init__(self):
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> bool:
        if self.httpd is not None:
            return True
        for _ in range(30):                    # port may be draining
            try:
                self.httpd = ThreadingHTTPServer(("0.0.0.0", PORT),
                                                 _Placeholder)
                break
            except OSError:
                if healthy():
                    return False               # the app already owns it
                time.sleep(1)
        if self.httpd is None:
            log("WARN: placeholder could not bind the port")
            return False
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        log(f"placeholder serving on :{PORT}")
        return True

    def stop(self):
        """Synchronous: returns only when the port is actually free
        (uvicorn cannot bind while the placeholder holds it)."""
        if self.httpd is not None:
            httpd, self.httpd = self.httpd, None
            httpd.shutdown()
            httpd.server_close()
            if self.thread:
                self.thread.join(timeout=3)


# ── provisioning ────────────────────────────────────────────────────
def _req_hash() -> str:
    try:
        with open(os.path.join(ROOT, "requirements.txt"), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return "none"


def _run(cmd, timeout_s, logfile=None):
    log("+ " + " ".join(str(c) for c in cmd))
    out = open(logfile, "ab") if logfile else None
    try:
        r = subprocess.run(cmd, timeout=timeout_s, stdout=out or None,
                           stderr=subprocess.STDOUT if out else None)
        return r.returncode
    except subprocess.TimeoutExpired:
        log(f"WARN: step timed out after {timeout_s}s: {cmd[0]}")
        return 124
    finally:
        if out:
            out.close()


def ensure_env() -> str:
    """Create venv + install core requirements when stale.
    Returns the interpreter to run the app with."""
    os.makedirs(os.path.join(DATA, "history"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    settings = os.path.join(DATA, "settings.json")
    if not os.path.exists(settings):
        with open(settings, "w") as fh:
            fh.write('{"theme":"dark","haptics":true,"sound":true}')
    if not os.path.exists(VENV_PY):
        _state["stage"] = "creating virtualenv"
        _run([sys.executable, "-m", "venv", ".venv"], 300)
    py = VENV_PY if os.path.exists(VENV_PY) else sys.executable
    want = _req_hash()
    have = ""
    if os.path.exists(STAMP):
        have = open(STAMP).read().split(" ")[0].strip()
    if os.environ.get("SKIP_SETUP") == "1" or (have == want
                                               and py == VENV_PY):
        log("environment fresh — skipping provisioning")
        return py
    _state["stage"] = "downloading core requirements (pip)"
    rc = _run([py, "-m", "pip", "install", "-q",
               "--disable-pip-version-check", "--no-input",
               "--retries", "2", "--timeout", "30",
               "-r", "requirements.txt"], 900)
    if rc != 0:
        log(f"WARN: pip exited {rc} — attempting engine start anyway")
    else:
        log("core requirements installed ✓")
    with open(STAMP, "w") as fh:
        fh.write(f"{want} {int(time.time())}\n")
    return py


def git_update() -> None:
    """Best-effort ff-pull before first launch (bounded, silent on
    any problem; a dirty tree is left untouched and the blocker is
    logged — generated docs.html dirt is discarded first)."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        dirty = subprocess.run(["git", "diff", "--name-only"],
                               capture_output=True, text=True,
                               timeout=10, env=env).stdout.split()
        if dirty and all(f == "docs.html" for f in dirty):
            subprocess.run(["git", "checkout", "--", "docs.html"],
                           timeout=10, env=env)
            dirty = []
        if dirty:
            log(f"auto-update skipped — dirty tree: {' '.join(dirty)}")
            return
        status = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True,
                                timeout=10, env=env).stdout.strip()
        if status:
            log("auto-update skipped — untracked/staged changes present")
            return
        before = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                timeout=10, env=env).stdout.strip()
        r = subprocess.run(["git", "pull", "-q", "--ff-only"],
                           capture_output=True, text=True,
                           timeout=30, env=env)
        after = subprocess.run(["git", "rev-parse", "HEAD"],
                               capture_output=True, text=True,
                               timeout=10, env=env).stdout.strip()
        if r.returncode == 0 and before != after:
            log(f"code updated {before[:7]} -> {after[:7]}")
    except Exception as e:
        log(f"auto-update skipped ({e.__class__.__name__})")


def editor_extras() -> None:
    """Best-effort editor niceties in a daemon thread: markdown
    preview extensions (bounded), docs.html regen. Never gates
    the boot, never touches the app port."""
    def work():
        try:
            subprocess.run(["timeout", "20", "code", "--list-extensions"],
                           capture_output=True, timeout=30)
        except Exception:
            pass
        try:
            have = subprocess.run(["timeout", "20", "code",
                                   "--list-extensions"],
                                  capture_output=True, text=True,
                                  timeout=30).stdout.lower()
            for ext in ("shd101wyy.markdown-preview-enhanced",
                        "yzhang.markdown-all-in-one"):
                if ext not in have:
                    subprocess.run(["timeout", "120", "code",
                                    "--install-extension", ext],
                                   capture_output=True, timeout=150)
        except Exception:
            pass
        try:
            subprocess.run([sys.executable,
                            "scripts/ensure_md_keybindings.py"],
                           capture_output=True, timeout=30)
        except Exception:
            pass
        try:
            subprocess.run([sys.executable, "scripts/make_docs_html.py"],
                           capture_output=True, timeout=60)
        except Exception:
            pass
    threading.Thread(target=work, daemon=True).start()


def make_port_public() -> None:
    """Port must ALWAYS be public (user order). NOTE, verified
    against the containers.dev spec: portsAttributes has NO
    "visibility" property — Codespaces ignores it, so the ONLY
    way to force public is the gh CLI. Retry for ~3 minutes:
    the forwarded port may not be registered yet when the
    placeholder first binds, and a single early attempt failing
    silently is exactly how a Codespace stays private. Also
    re-asserted once the app is healthy (serve loop)."""
    name = os.environ.get("CODESPACE_NAME")
    if not name:
        return
    def attempt() -> bool:
        try:
            r = subprocess.run(
                ["gh", "codespace", "ports", "visibility",
                 f"{PORT}:public", "-c", name],
                capture_output=True, text=True, timeout=25)
            if r.returncode == 0:
                return True
            log(f"port visibility attempt failed (gh {r.returncode}): "
                f"{(r.stderr or r.stdout or '').strip()[:160]}")
        except Exception as e:
            log(f"port visibility attempt skipped "
                f"({e.__class__.__name__})")
        return False
    def work():
        time.sleep(2)
        for i in range(6):
            if attempt():
                log("port visibility -> public ✓")
                return
            time.sleep(30)
        log("WARN: could not force port public after 6 attempts — "
            "set it once via Ports tab > Port Visibility > Public")
    threading.Thread(target=work, daemon=True).start()


# ── engine child ────────────────────────────────────────────────────
def launch_engine(py: str) -> subprocess.Popen:
    out = open(SERVERLOG, "ab")
    proc = subprocess.Popen(
        [py, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0",
         "--port", str(PORT), "--workers", "1"],
        stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"uvicorn launched (pid {proc.pid}) — log: data/server.log")
    return proc


def stop_engine(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def serve() -> int:
    def _term(signum, frame):
        stop_engine(_child)
        try:
            os.unlink(PIDFILE)
        except OSError:
            pass
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    os.makedirs(DATA, exist_ok=True)
    with open(PIDFILE, "w") as fh:
        fh.write(str(os.getpid()))
    placeholder = _PlaceholderServer()
    placeholder.start()
    make_port_public()
    editor_extras()
    git_update()

    def _deferred_backtest():
        time.sleep(600)             # daily real backtest, long after
        _run(["bash", "scripts/run_backtest_once.sh"], 3600)   # boot
    threading.Thread(target=_deferred_backtest, daemon=True).start()
    py = ensure_env()

    global _child
    fails = 0
    while True:
        _state["stage"] = "checking engine import"
        rc = _run([py, "-c", "import app.main"], 120, logfile=SERVERLOG)
        if rc != 0:
            _state["stage"] = "engine import failing (see server log)"
            _state["detail"] = "the placeholder stays up until the code imports"
            log("import failed — retrying in 30 s (placeholder stays up)")
            time.sleep(30)
            continue
        _state["stage"] = "starting engine"
        _state["detail"] = "uvicorn is booting"
        placeholder.stop()          # free the port BEFORE uvicorn binds
        _child = launch_engine(py)
        deadline = time.time() + 120
        while time.time() < deadline:
            if _child.poll() is not None:
                log(f"engine exited rc={_child.returncode} during boot")
                break
            if healthy():
                break
            time.sleep(1)
        if healthy():
            _state["stage"] = "live"
            log(f"✓ app healthy on http://0.0.0.0:{PORT}")
            make_port_public()      # re-assert public on the real app
            fails = 0
        else:
            stop_engine(_child)
            _child = None
            placeholder.start()
            _state["stage"] = "engine failed to boot (see server log)"
            log("engine not healthy — placeholder restored, retry in 10 s")
            time.sleep(10)
            continue
        # ── watchdog: child alive + health, else restart ──
        while True:
            time.sleep(5)
            if _child.poll() is not None:
                log(f"engine died rc={_child.returncode} — restarting")
                placeholder.start()
                break
            if healthy():
                fails = 0
            else:
                fails += 1
                if fails >= 12:                # ~60 s of bad health
                    log("engine hung (60 s unhealthy) — restarting")
                    stop_engine(_child)
                    placeholder.start()
                    break
        time.sleep(3)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_pid() -> int | None:
    try:
        pid = int(open(PIDFILE).read().strip())
        return pid if _pid_alive(pid) else None
    except (OSError, ValueError):
        return None


def detach() -> int:
    if healthy():
        print("app already healthy — nothing to do", flush=True)
        return 0
    pid = running_pid()
    if pid:
        print(f"starter already running (pid {pid})", flush=True)
        return 0
    os.makedirs(DATA, exist_ok=True)
    out = open(BOOTLOG, "ab")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                             "--serve"], stdout=out, stderr=out,
                            start_new_session=True)
    print(f"starter launched (pid {proc.pid}) — port {PORT} is "
          f"covered by the placeholder within seconds", flush=True)
    return 0


def stop() -> int:
    pid = running_pid()
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        print(f"stopped starter pid {pid}", flush=True)
    else:
        print("no starter running", flush=True)
    return 0


def status() -> int:
    if healthy():
        print(f"port {PORT}: app HEALTHY")
    elif http_ok(f"http://127.0.0.1:{PORT}/healthz", 2):
        print(f"port {PORT}: placeholder (app still starting)")
    else:
        print(f"port {PORT}: NOTHING LISTENING (run: python3 start.py --detach)")
    pid = running_pid()
    print(f"starter pid: {pid or 'none'}")
    return 0


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "--detach"
    if arg == "--detach":
        return detach()
    if arg == "--serve":
        try:
            return serve()
        except KeyboardInterrupt:
            stop_engine(_child)
            return 0
    if arg == "--stop":
        return stop()
    if arg == "--status":
        return status()
    if arg in ("--foreground", "-f"):
        try:
            return serve()
        except KeyboardInterrupt:
            stop_engine(_child)
            return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
