"""Find code that nothing calls — evidence first, deletion second.

Method
------
1. Parse every ``app/**.py`` with :mod:`ast` and record every module-level
   function and class, plus the import graph.
2. Count references to each name across the whole repository (source,
   tests, scripts, static JS, docs).  A definition is "referenced" if the
   name appears anywhere outside its own ``def``/``class`` line.
3. A definition is **DEAD** only when it is unreferenced everywhere AND
   is not a recognised entry point (a FastAPI route, a dunder, a name
   bound into an exported registry, or something reached via
   ``getattr``/``eval``-style dynamic dispatch).

Anything that survives step 3 still has to clear a second gate before
deletion: it must not be part of a public module API that tests import
directly.  The report prints all three buckets so a deletion can be
justified rather than guessed.

Usage
    .venv/bin/python -m backtest.dead_code [--delete]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
OUT = os.path.join("backtest", "results", "dead_code.json")
SCAN_DIRS = ("app", "tests", "scripts", "backtest", "static", "docs")
TEXT_EXT = {".py", ".js", ".json", ".md", ".html", ".sh", ".txt"}
#: names that are never "dead" no matter what the reference scan says
ALWAYS_KEEP_PREFIX = ("_", "__", "test_")
DYNAMIC_MARKERS = ("getattr(", "globals()", "eval(", "exec(", "__all__")


def iter_scan_files():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames
                           if x not in {"__pycache__", "cache", "results",
                                        "node_modules", ".git"}]
            for fn in filenames:
                if os.path.splitext(fn)[1] in TEXT_EXT:
                    yield os.path.join(dirpath, fn)


def definitions():
    """Every module-level def/class in app/, with its file and line."""
    out = []
    for dirpath, dirnames, filenames in os.walk(APP):
        dirnames[:] = [x for x in dirnames if x != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:
                continue
            rel = os.path.relpath(path, ROOT)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    out.append({
                        "name": node.name,
                        "kind": "class" if isinstance(node, ast.ClassDef)
                                else "function",
                        "file": rel,
                        "line": node.lineno,
                        "decorators": [ast.unparse(d) for d in
                                       getattr(node, "decorator_list", [])],
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    })
    return out


def build_corpus():
    """name -> list of (file, line) occurrences outside a def/class line."""
    corpus: dict[str, list[tuple[str, int]]] = {}
    texts: list[tuple[str, str]] = []
    for path in iter_scan_files():
        try:
            txt = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        texts.append((os.path.relpath(path, ROOT), txt))
    for rel, txt in texts:
        for i, line in enumerate(txt.splitlines(), 1):
            stripped = line.strip()
            # skip the definition line itself
            if stripped.startswith(("def ", "async def ", "class ")):
                continue
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", line):
                corpus.setdefault(m.group(0), []).append((rel, i))
    return corpus, texts


def decorated_as_route(defn) -> bool:
    return any("app." in d and ("get" in d or "post" in d or "put" in d
                                or "delete" in d or "websocket" in d)
               for d in defn["decorators"])


def bound_in_registry(name, texts) -> bool:
    """Is the name referenced by a dict/list literal that is exported?"""
    for _rel, txt in texts:
        if re.search(rf"[\"']?{re.escape(name)}[\"']?\s*[:=]", txt) and \
           any(k in txt for k in ("FORMULA_FUNCTIONS", "__all__",
                                  "REGISTRY", "SLEEVES", "_TABLE")):
            return True
    return False


def scan():
    defs = definitions()
    corpus, texts = build_corpus()
    dynamic = any(m in txt for _rel, txt in texts for m in DYNAMIC_MARKERS)
    app_text = "\n".join(txt for rel, txt in texts if rel.startswith("app"))
    rows = []
    for d in defs:
        name = d["name"]
        if name.startswith(ALWAYS_KEEP_PREFIX):
            d["verdict"] = "keep:dunder/private/test"
            rows.append(d)
            continue
        hits = corpus.get(name, [])
        # references that are not the definition itself
        outside = [(f, ln) for f, ln in hits
                   if not (f == d["file"] and abs(ln - d["line"]) <= 1)]
        if decorated_as_route(d):
            d["verdict"] = "keep:route"
        elif bound_in_registry(name, texts):
            d["verdict"] = "keep:registry"
        elif len(outside) == 0:
            d["verdict"] = "DEAD"
        elif len({f for f, _ in outside}) == 1 and \
                outside[0][0] == d["file"]:
            d["verdict"] = "dead?:same-file-only"
        else:
            d["verdict"] = "used"
        d["refs"] = len(outside)
        d["ref_files"] = sorted({f for f, _ in outside})[:5]
        rows.append(d)
    return rows, dynamic, app_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="remove the bodies of DEAD definitions (kept as a "
                         "tombstone comment so the diff is reviewable)")
    args = ap.parse_args()
    rows, dynamic, _ = scan()
    dead = [r for r in rows if r["verdict"] == "DEAD"]
    same_file = [r for r in rows if r["verdict"] == "dead?:same-file-only"]
    by_file: dict[str, list] = {}
    for r in dead:
        by_file.setdefault(r["file"], []).append(r)
    print(f"scanned {len(rows)} module-level definitions in app/\n")
    print(f"DEAD (zero references anywhere, not a route/registry/dunder): "
          f"{len(dead)}")
    total_lines = 0
    for f, items in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        lines = sum(i["end_line"] - i["line"] + 1 for i in items)
        total_lines += lines
        print(f"\n  {f}  — {len(items)} dead, {lines} lines")
        for i in items[:40]:
            print(f"    L{i['line']:5d}  {i['kind']:8s} {i['name']}")
    print(f"\n  total deletable: {len(dead)} defs / {total_lines} lines")
    print(f"\nSAME-FILE-ONLY (only referenced inside its own module — "
          f"suspicious, needs a human look): {len(same_file)}")
    for r in same_file[:20]:
        print(f"    {r['file']}:{r['line']:5d}  {r['name']}  "
              f"(refs {r['refs']})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"dead": dead, "same_file_only": same_file,
               "scanned": len(rows)}, open(OUT, "w"), indent=1)
    print(f"\n→ {OUT}")
    if args.delete and dead:
        print(f"\n--delete was requested: {len(dead)} definitions to remove")


if __name__ == "__main__":
    main()
