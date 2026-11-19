"""The dashboard must survive the payload it is actually given.

Static greps cannot catch the bug that shipped here: ``loadEmotion`` read
``fg.unmeasured.join(", ")`` while the engine sends a *name → reason* map, and
``loadAgents`` read ``a.local.install`` while the installer is exposed as a
top-level ``a.installer``. Both render ``undefined`` (or throw) in a real
browser, and both would pass any test that only asserts a string appears in
``app.js``.

So these tests EXTRACT the rendering block out of ``static/app.js`` verbatim
and execute it under Node against a real payload. If the JS and the API ever
disagree again, this fails instead of the panel going blank.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_JS = ROOT / "static" / "app.js"

# pytest is run both from the repo root and from tests/ in this project, so
# make `import app.*` work regardless of which one we were started in.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is needed to execute the render block")

PRELUDE = r"""
const fmt = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? "\u2014" : Number(v).toFixed(n);
const esc = (v) => String(v ?? "").replace(/&/g, "&amp;")
  .replace(/</g, "&lt;").replace(/>/g, "&gt;");
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const api = async () => ({});
const toast = () => {};
"""


def _node(script: str) -> str:
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                          timeout=120, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _slice(start_marker: str, end_marker: str) -> str:
    src = APP_JS.read_text()
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    return src[i:j]


def _emotion_payload() -> dict:
    """A real engine payload — never a hand-written stub.

    Driving the real engine is the whole point: a stub would just repeat
    whatever shape this test already believes the API has.
    """
    from test_engine import make_engine, run, _boot
    eng, _f, _e = make_engine()
    run(_boot(eng))
    eng.compute_asset("BTC", record=False, display=True)
    return eng.build_snapshot()["systems"]["BTC|tf1"]["emotion"]


def _render_fear_greed(em: dict) -> str:
    body = _slice('        const fg = em.fear_greed || null;',
                  '        const ca = em.core_affect') + "\nreturn fgHtml;"
    script = (PRELUDE
              + "const em = " + json.dumps(em) + ";\n"
              + "const out = (function () {\n" + body + "\n})();\n"
              + 'console.log(out.replace(/<[^>]+>/g, " ").replace(/\\s+/g, " ").trim());')
    return _node(script)


def test_fear_greed_renders_partial_composite_without_throwing():
    em = _emotion_payload()
    assert em["fear_greed"]["partial"] is True
    assert isinstance(em["fear_greed"]["unmeasured"], dict)

    out = _render_fear_greed(em)
    assert "fear" in out.lower()
    # every unmeasured component must be NAMED, not iterated as characters
    for name in em["fear_greed"]["unmeasured"]:
        assert name in out.lower(), f"{name} missing from the panel text: {out}"
    assert "partial" in out.lower()
    assert "%" in out
    # and it must not leak the raw map/array rendering
    assert "[object Object]" not in out
    assert "undefined" not in out


def test_fear_greed_absent_renders_nothing_rather_than_undefined():
    em = dict(_emotion_payload())
    em["fear_greed"] = None
    assert _render_fear_greed(em) == ""


def test_agents_panel_reads_the_top_level_installer():
    """The installer is ``agents.installer``, not ``agents.local.install``.

    Reading the wrong path left the whole local-agent panel undefined — which
    is why the app looked like it was "not downloading" even while it was.
    """
    payload = {
        "installer": {
            "state": "failed", "phase": "model", "backend": "llama_cpp",
            "installed": False, "model_present": False,
            "model_label": "Qwen2-0.5B-Instruct (Q4_K_M)",
            "model_size": "~400 MB", "bytes_on_disk": 0,
            "ollama": {"ok": False, "detail": "ConnectError", "models": []},
            "compiler": {"ok": True, "cxx": "g++"},
            "steps": [{"step": "llama_cpp", "ok": True, "reason": "0.3.37"},
                      {"step": "model", "ok": False, "reason": "hub unreachable"}],
            "log": ["line one"], "error": "no model file",
            "auto": {"state": "failed", "attempts": 2,
                     "last_error": "hub unreachable"},
        },
        "local": {"available": False, "label": "unavailable", "ram_gb": 3.8},
    }
    body = _slice('    const lo = a.local || {};', '    const autoRunning =')
    body = body.replace('    const lo = a.local || {};',
                        '    const lo = a.local || {};', 1)
    script = (PRELUDE
              + "const a = " + json.dumps(payload) + ";\n"
              + 'let captured = "";\n'
              + 'const $ = (sel) => ({ set innerHTML(v) '
                '{ if (sel === "#ag-local") captured = v; },\n'
              + '                      addEventListener() {}, disabled: false, '
                'textContent: "" });\n'
              + body + "\n"
              + 'console.log(captured.replace(/<[^>]+>/g, " ").replace(/\\s+/g, " ").trim());')
    out = _node(script)
    for needle in ("llama_cpp", "hub unreachable", "ConnectError", "g++",
                   "Qwen2-0.5B-Instruct", "re-probe runtime"):
        assert needle.lower() in out.lower(), f"{needle!r} not rendered: {out[:500]}"
    assert "undefined" not in out
    assert "[object Object]" not in out


def test_retry_route_is_wired_to_a_visible_button():
    js = APP_JS.read_text()
    assert 'api("/api/agents/local/retry"' in js
    assert 'id="ag-retry"' in js
