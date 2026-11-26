#!/usr/bin/env python3
"""Mechanical deep audit of app/ and static/.

Checks EVERY file for a known list of defect classes, so the pass does not
depend on a human (or an agent) remembering to look. Every check corresponds
to a bug class already found in this repo, or a standard correctness hazard.

Precedents:
  sign-inversion    cci.py:  cci*(1 - k*|rev|/|cci|) went negative and flipped
                    the trading direction (net-bearish vote issued a BUY)
  nan-to-max        cci.py:  max(0.02, min(0.98, nan)) == 0.98, so a failed
                    calculation displayed as maximum confidence
  ungraded-counted  ledger.py: rows with barriers_available=False still stored
                    result "win"/"loss" and moved the published win rate
  missing-to-zero   app.js:  pct(NaN) rendered "0.00%", a missing value shown
                    as a measured zero
  lookahead         beat_funds: np.convolve(mode="same") is centred and read
                    future bars (fake t = 47)

Run:
    .venv/bin/python scripts/deep_audit.py
    .venv/bin/python scripts/deep_audit.py --verbose
    .venv/bin/python scripts/deep_audit.py --json
    .venv/bin/python scripts/deep_audit.py --check silent-except

Exit code 0 when there are no HIGH findings, 1 otherwise (CI gate).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
STATIC = os.path.join(ROOT, "static")

# Running `python scripts/deep_audit.py` puts scripts/ on sys.path, not the
# repo root — so `import app.barriers` failed and the function probe silently
# reported 0 functions. This tool must never fail silently about itself.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# An imported app module closes file descriptor 1 (measured: probing killed
# every later print with OSError EBADF, which looked like "no findings").
# Duplicate stdout NOW, before any app module is imported, and write results
# to that duplicate at the end.
try:
    _STDOUT_FD = os.dup(1)
except OSError:
    _STDOUT_FD = None

HIGH, MED, LOW = "HIGH", "MED", "LOW"


@dataclass
class Finding:
    severity: str
    check: str
    path: str
    line: int
    message: str
    snippet: str = ""


@dataclass
class Report:
    findings: list = field(default_factory=list)
    files_scanned: int = 0
    functions_probed: int = 0

    def add(self, *a, **kw):
        self.findings.append(Finding(*a, **kw))

    def by_severity(self, sev):
        return [f for f in self.findings if f.severity == sev]


def _rel(p):
    return os.path.relpath(p, ROOT)


def _py_files():
    for dirpath, dirnames, filenames in os.walk(APP):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _js_files():
    if not os.path.isdir(STATIC):
        return
    for dirpath, dirnames, filenames in os.walk(STATIC):
        dirnames[:] = [d for d in dirnames if d != "vendor"]
        for name in sorted(filenames):
            if name.endswith(".js"):
                yield os.path.join(dirpath, name)


#: name fragments that imply I/O, process control, or mutation
_SIDE_EFFECT = re.compile(
    r"download|install|fetch|run|start|stop|kill|exec|spawn|launch|serve|boot|"
    r"write|save|load|open|read|delete|remove|unlink|connect|close|send|post|"
    r"get_|poll|update|set_|reset|clear|flush|emit|apply|commit|restart|"
    r"submit|raise_|trigger|sync|patch|register|init|build|compile|train|"
    r"ensure|provision|verify|check_|probe|watch|monitor|loop|stream|"
    r"pull|push|upload|migrate|seed|warm|repair|purge|vacuum|gc_",
    re.IGNORECASE)

BUILTINS = {
    "sum", "min", "max", "abs", "round", "len", "list", "dict", "set", "tuple",
    "str", "int", "float", "bool", "bytes", "type", "id", "hash", "map",
    "filter", "sorted", "all", "any", "next", "iter", "open", "input", "print",
}


def check_python(path, report):
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as exc:
        report.add(HIGH, "syntax-error", _rel(path), exc.lineno or 0,
                   "file does not parse: %s" % exc.msg)
        return
    report.files_scanned += 1
    lines = src.splitlines()

    def snip(n):
        return lines[n - 1].strip()[:160] if 1 <= n <= len(lines) else ""

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + \
                [d for d in node.args.kw_defaults if d is not None]
            for default in defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    report.add(HIGH, "mutable-default", _rel(path), node.lineno,
                               "%s() has a mutable default argument (shared "
                               "across calls)" % node.name, snip(node.lineno))
                elif isinstance(default, ast.Call) and \
                        isinstance(default.func, ast.Name) and \
                        default.func.id in {"list", "dict", "set"}:
                    report.add(HIGH, "mutable-default", _rel(path), node.lineno,
                               "%s() default is a mutable call result" % node.name,
                               snip(node.lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body = node.body
            silent = len(body) == 1 and isinstance(body[0], ast.Pass)
            if node.type is None:
                report.add(HIGH if silent else MED, "bare-except", _rel(path),
                           node.lineno,
                           "bare `except:` " + ("swallows everything silently"
                                                if silent else
                                                "catches KeyboardInterrupt/SystemExit"),
                           snip(node.lineno))
            elif silent:
                report.add(MED, "silent-except", _rel(path), node.lineno,
                           "`except ...: pass` - a failure here is invisible",
                           snip(node.lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and \
                        isinstance(comp, ast.Constant) and \
                        isinstance(comp.value, float):
                    report.add(MED, "float-equality", _rel(path), node.lineno,
                               "exact float comparison against %r" % (comp.value,),
                               snip(node.lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
            rhs = node.right
            guarded = isinstance(rhs, ast.Constant) and rhs.value not in (0, 0.0)
            if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) \
                    and rhs.func.id in {"max", "abs"}:
                guarded = True
            if not guarded and isinstance(rhs, ast.Name):
                window = "\n".join(lines[max(0, node.lineno - 3):node.lineno])
                if not re.search(r"(>|>=|!=)\s*0|or\s+1e-|max\(|isfinite", window):
                    report.add(LOW, "unguarded-division", _rel(path), node.lineno,
                               "division by `%s` with no visible guard" % rhs.id,
                               snip(node.lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "math":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, float) \
                        and not math.isfinite(arg.value):
                    report.add(HIGH, "math-nonfinite", _rel(path), node.lineno,
                               "math.%s() called with a non-finite literal"
                               % node.func.attr, snip(node.lineno))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in node.args.args + node.args.kwonlyargs:
                if a.arg in BUILTINS:
                    report.add(LOW, "shadowed-builtin", _rel(path), node.lineno,
                               "%s() parameter `%s` shadows a builtin"
                               % (node.name, a.arg), snip(node.lineno))

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"1(\.0)?\s*-\s*[\w.]+\s*\*.*/\s*abs\(", line):
            report.add(MED, "possible-sign-inversion", _rel(path), i,
                       "multiplicative (1 - k*x/|y|) penalty can go negative and "
                       "invert the sign - clamp with max(0, .)", stripped[:160])
        if "convolve" in line and "same" in line:
            report.add(MED, "lookahead-convolve", _rel(path), i,
                       "np.convolve(..., mode='same') is a CENTRED window - on a "
                       "time series this reads future bars", stripped[:160])
        if re.search(r"sorted\([^)]*\)\[0\]", line):
            report.add(MED, "empty-index", _rel(path), i,
                       "sorted(...)[0] raises IndexError on an empty sequence",
                       stripped[:160])


def _nonfinite_deep(v, depth=0):
    if depth > 3:
        return None
    if isinstance(v, float):
        return None if math.isfinite(v) else v
    if isinstance(v, dict):
        for k, x in v.items():
            r = _nonfinite_deep(x, depth + 1)
            if r is not None:
                return {k: r}
    if isinstance(v, (list, tuple)):
        for x in v:
            r = _nonfinite_deep(x, depth + 1)
            if r is not None:
                return r
    return None


def probe_python(path, report):
    import importlib
    import inspect
    rel = os.path.relpath(path, ROOT)
    modname = rel[:-3].replace(os.sep, ".")
    try:
        mod = importlib.import_module(modname)
    except Exception as exc:
        # never hide our own blindness: an unimportable module is a finding
        report.add(MED, "probe-import-failed", _rel(path), 0,
                   "could not import %s (%s) - probe skipped"
                   % (modname, type(exc).__name__))
        return
    EDGE = [0.0, -0.0, 1.0, -1.0, 1e-12, -1e-12, 1e12,
            float("nan"), float("inf"), -float("inf"), None, True, False, 0, 1, -1]
    for name, fn in list(vars(mod).items()):
        if name.startswith("_") or not callable(fn):
            continue
        if getattr(fn, "__module__", None) != modname:
            continue
        if inspect.isclass(fn) or not inspect.isfunction(fn):
            continue            # classes/partials have no __code__ to report on
        if inspect.iscoroutinefunction(fn):
            continue            # probing one would only create an unawaited coroutine
        # NEVER PROBE ANYTHING WITH SIDE EFFECTS.
        # Measured the hard way: the probe called agents.download_gguf(), which
        # spawned a real installer process and closed file descriptor 1, so the
        # audit appeared to produce no output at all. Only pure computations are
        # safe to call with arbitrary garbage.
        if _SIDE_EFFECT.search(name):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = [p for p in sig.parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                  and p.default is p.empty]
        if not params or len(params) > 3:
            continue
        report.functions_probed += 1
        for va in EDGE:
            args = (va,) * len(params)
            try:
                out = fn(*args)
            except Exception as exc:
                if isinstance(exc, (ZeroDivisionError, TypeError, ValueError,
                                    OverflowError, IndexError, KeyError)):
                    report.add(LOW, "probe-raises", _rel(path),
                               getattr(getattr(fn, "__code__", None),
                                       "co_firstlineno", 0),
                               "%s%s raises %s" % (name, args[:len(params)],
                                                   type(exc).__name__))
                continue
            bad = _nonfinite_deep(out)
            if bad:
                report.add(HIGH, "probe-nonfinite", _rel(path),
                           getattr(getattr(fn, "__code__", None),
                                   "co_firstlineno", 0),
                           "%s%s -> non-finite %r" % (name, args[:len(params)], bad))


def check_js(path, report):
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    report.files_scanned += 1
    for i, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if s.startswith("//") or s.startswith("*"):
            continue
        if re.search(r"\?\?\s*0\b|\|\|\s*0\b", line) and "toFixed" in line:
            report.add(MED, "missing-to-zero", _rel(path), i,
                       "`|| 0` before toFixed() renders a MISSING value as a "
                       "measured zero", s[:160])
        for m in re.finditer(r"\.toFixed\(", line):
            pre = line[:m.start()]
            if re.search(r"Number\(|parseFloat|\bfx\(|\bfmt\(|_unknown|\.toFixed\(k", pre):
                continue
            report.add(LOW, "unguarded-tofixed", _rel(path), i,
                       "toFixed() on an unguarded value renders NaN", s[:160])
        if "innerHTML" in line and "${" in line and "esc(" not in line \
                and "fx(" not in line and "px(" not in line:
            report.add(LOW, "unescaped-html", _rel(path), i,
                       "innerHTML interpolation without esc()/fx()", s[:160])


def _flush(lines):
    text = ("\n".join(lines) + "\n").encode("utf-8", "replace")
    if _STDOUT_FD is not None:
        try:
            os.write(_STDOUT_FD, text)
            return
        except OSError:
            pass
    try:
        sys.stderr.write(text.decode("utf-8", "replace"))
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def main():
    # Some imported app modules reconfigure fd 1 (the JSON-safety logger and
    # the feed both touch stdout). Probing them left this tool unable to print,
    # which looked like "no findings". Buffer our own output and emit at the
    # end through whatever stdout is left, or stderr if it is gone.
    out_lines = []

    def emit(text=""):
        out_lines.append(text)

    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--check")
    args = ap.parse_args()

    report = Report()
    for path in _py_files():
        check_python(path, report)
        if not args.no_probe:
            probe_python(path, report)
    for path in _js_files():
        check_js(path, report)

    findings = report.findings
    if args.check:
        findings = [f for f in findings if f.check == args.check]

    if args.json:
        emit(json.dumps({"files_scanned": report.files_scanned,
                         "functions_probed": report.functions_probed,
                         "findings": [asdict(f) for f in findings]}, indent=2))
        _flush(out_lines)
        return 0 if not [f for f in findings if f.severity == HIGH] else 1

    order = {HIGH: 0, MED: 1, LOW: 2}
    counts = {}
    for f in findings:
        counts[(f.severity, f.check)] = counts.get((f.severity, f.check), 0) + 1

    emit("scanned %d files, probed %d functions"
         % (report.files_scanned, report.functions_probed))
    emit("findings: %d (HIGH %d, MED %d, LOW %d)\n"
         % (len(findings), len(report.by_severity(HIGH)),
            len(report.by_severity(MED)), len(report.by_severity(LOW))))

    if counts:
        emit("%-10s%-30s%6s" % ("severity", "check", "count"))
        for (sev, check), n in sorted(counts.items(),
                                      key=lambda kv: (order[kv[0][0]], -kv[1])):
            emit("%-10s%-30s%6d" % (sev, check, n))
        emit()

    shown = findings if args.verbose else \
        [f for f in findings if f.severity in (HIGH, MED)]
    for f in sorted(shown, key=lambda f: (order[f.severity], f.path, f.line)):
        emit("[%s] %s:%d  %s" % (f.severity, f.path, f.line, f.check))
        emit("       %s" % f.message)
        if f.snippet:
            emit("       | %s" % f.snippet)

    if not args.verbose and len(shown) != len(findings):
        emit("\n(%d LOW findings hidden; rerun with --verbose)"
             % (len(findings) - len(shown)))
    return 0 if not report.by_severity(HIGH) else 1


if __name__ == "__main__":
    sys.exit(main())
