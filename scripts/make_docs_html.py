#!/usr/bin/env python3
"""Render README.md + API_SETUP.md into a self-contained docs.html.

Why: some editors ship WITHOUT built-in markdown, so the preview button
errors with "markdown.showPreview not found" and no extension can fix it.
docs.html opens with a double-click in any browser — zero extensions.
"""
from __future__ import annotations
import html as _h
import os
import re as _re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(text: str) -> str:
    """Minimal safe markdown -> HTML (same engine as the /docs route)."""
    out, in_code, in_ul, in_tbl = [], False, False, False

    def close_lists():
        nonlocal in_ul, in_tbl
        if in_ul:
            out.append("</ul>"); in_ul = False
        if in_tbl:
            out.append("</table>"); in_tbl = False

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            close_lists()
            out.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(_h.escape(line)); continue
        if not line.strip():
            close_lists(); continue
        esc = _h.escape(line)
        esc = _re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
        esc = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", esc)
        esc = _re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", esc)
        esc = _re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', esc)
        m = _re.match(r"^(#{1,6})\s+(.*)$", esc)
        if m:
            close_lists(); lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl}>{m.group(2)}</h{lvl}>"); continue
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(_re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            if not in_tbl:
                out.append("<table>"); in_tbl = True
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>"); in_tbl = False
        m = _re.match(r"^\s*[-*+]\s+(.*)$", esc)
        if m:
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{m.group(1)}</li>"); continue
        if in_ul:
            out.append("</ul>"); in_ul = False
        out.append(f"<p>{esc}</p>")
    close_lists()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def main() -> int:
    parts = []
    for name in ("README.md", "API_SETUP.md"):
        fp = os.path.join(ROOT, name)
        try:
            with open(fp, encoding="utf-8", errors="replace") as fh:
                parts.append(f'<article><h1 class="doc-title">{name}</h1>{md(fh.read())}</article>')
        except OSError:
            parts.append(f"<article><p>{name} not found</p></article>")
    page = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PREDICT · docs</title><style>
body{background:#07070d;color:#d8d8e8;font:14px/1.6 -apple-system,"Segoe UI",sans-serif;
     max-width:900px;margin:0 auto;padding:40px 24px}
h1,h2,h3,h4,h5,h6{color:#c9a227;font-family:ui-monospace,monospace;letter-spacing:.5px;margin:1.4em 0 .5em}
h1.doc-title{border-bottom:1px solid #1d1d2e;padding-bottom:8px;font-size:20px}
pre{background:#0e0e18;border:1px solid #1d1d2e;border-radius:8px;padding:14px;overflow-x:auto}
code{font-family:ui-monospace,monospace;background:#12121f;padding:1px 5px;border-radius:3px;font-size:12.5px}
a{color:#00d4f0} table{border-collapse:collapse;margin:12px 0;width:100%}
th,td{border:1px solid #1d1d2e;padding:6px 12px;font-size:12.5px} th{color:#6e6e8c}
ul{padding-left:22px} li{margin:4px 0} i{color:#9aa}
a.back{display:inline-block;margin-bottom:20px;color:#c9a227}
.banner{background:#14120a;border:1px solid #c9a227;border-radius:8px;padding:10px 14px;
  margin-bottom:20px;font-size:12.5px;color:#c9a227}
</style></head><body>
<div class="banner">📄 Static docs viewer — works with ZERO editor extensions.
Live version: <a href="/docs">http://localhost:8000/docs</a> · refresh this file
after doc changes with <code>python3 scripts/make_docs_html.py</code></div>
<a class="back" href="/">← back to app</a>""" + "".join(parts) + "</body></html>"
    with open(os.path.join(ROOT, "docs.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote docs.html ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
