#!/usr/bin/env python3
"""Fetch a REAL Drosophila connectome for the brain system.

Priority:
  1. neuPrint live API   (set NEUPRINT_URL + NEUPRINT_TOKEN — e.g.
     https://neuprint.janelia.org with a free token)
  2. GitHub mirror       (hemibrain v1.2 structural outputs, CC BY 4.0 —
     Janelia FlyEM data via github.com/ROHITCRAFTSYT/FruitFlyBrain)
  3. keep whatever is already in data/brain_real.json (or the structural
     fallback) — the app NEVER depends on this succeeding.

Writes data/brain_real.json: degree distribution of all 21,739 traced
neurons, top-40 hubs with real type/instance names (APL, DPM, MBONs…),
neurotransmitter census, graph statistics.
"""
from __future__ import annotations
import csv, io, json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "brain_real.json")
GH = "https://api.github.com/repos/ROHITCRAFTSYT/FruitFlyBrain/contents/"
FILES = {
    "degree": "projects/01_connectome_graph_analysis/outputs/neuron_degree_table.csv",
    "hubs": "projects/01_connectome_graph_analysis/outputs/top_hubs.csv",
    "summary": "projects/01_connectome_graph_analysis/outputs/summary.txt",
    "nt": "projects/02_neurotransmitter_atlas/outputs/counts_by_neurotransmitter.csv",
    "types": "projects/02_neurotransmitter_atlas/outputs/top_cell_types.csv",
}


def _gh(path: str) -> str:
    req = urllib.request.Request(GH + path,
                                 headers={"Accept": "application/vnd.github.raw"})
    return urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")


def fetch_neuprint() -> dict | None:
    url = os.environ.get("NEUPRINT_URL", "").rstrip("/")
    token = os.environ.get("NEUPRINT_TOKEN", "")
    if not url:
        return None
    try:
        import httpx
        r = httpx.get(url + "/api/metadata",
                      headers={"Authorization": f"Bearer {token}"} if token else {},
                      timeout=10.0)
        if r.status_code != 200:
            return None
        meta = r.json()
        return {"source": f'neuPrint live ({meta.get("meta", {}).get("databaseName", url)})',
                "via": url, "retrieved": "live", "stats": {
                    "neurons": meta.get("meta", {}).get("nodeCount"),
                    "edges": meta.get("meta", {}).get("edgeCount")}}
    except Exception as e:                                   # pragma: no cover
        print(f"neuPrint unreachable ({e}); trying GitHub mirror…")
        return None


def fetch_github() -> dict:
    degree = _gh(FILES["degree"])
    hubs = _gh(FILES["hubs"])
    summary = _gh(FILES["summary"])
    nt = _gh(FILES["nt"])
    types = _gh(FILES["types"])
    import re
    stats = {}
    for m in re.finditer(r"(neurons|edges|synapses|density|reciprocity)"
                         r"=([0-9.]+)", summary):
        k, v = m.group(1), m.group(2)
        if k in stats:          # first occurrence only (skip backbone line)
            continue
        stats[k] = float(v) if "." in v else int(v)
    degs = sorted((int(r["total_degree"]) for r in csv.DictReader(io.StringIO(degree))),
                  reverse=True)
    hub_rows = list(csv.DictReader(io.StringIO(hubs)))[:40]
    nts = {r["top_nt"]: int(r["n_neurons"]) for r in csv.DictReader(io.StringIO(nt))}
    ctypes = [[r["cell_type"], int(r["n_neurons"])]
              for r in csv.DictReader(io.StringIO(types))]
    return {
        "source": "hemibrain v1.2 — Janelia FlyEM (CC BY 4.0)",
        "via": "github.com/ROHITCRAFTSYT/FruitFlyBrain",
        "retrieved": __import__("datetime").date.today().isoformat(),
        "stats": stats,
        "degree_distribution": {
            "mean": round(sum(degs) / len(degs), 2),
            "median": degs[len(degs) // 2],
            "p90": degs[int(len(degs) * 0.10)],
            "p99": degs[int(len(degs) * 0.01)],
            "max": degs[0],
        },
        "top_degrees": degs[:5000],
        "hubs": [{"type": h["type"], "instance": h["instance"],
                  "total_degree": int(h["total_degree"]), "bodyId": h["bodyId"]}
                 for h in hub_rows],
        "neurotransmitters": nts,
        "cell_types": ctypes,
    }


def main() -> int:
    data = fetch_neuprint() or fetch_github()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh)
    print(f"real connectome saved → {OUT}")
    print(f"  source : {data.get('source')}")
    print(f"  stats  : {data.get('stats')}")
    print(f"  hubs   : {', '.join(h['type'] for h in (data.get('hubs') or [])[:6])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
