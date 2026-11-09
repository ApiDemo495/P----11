"""The four brains on REAL neurons (FlyWire v783).

Why this exists: DB-1..DB-4 used to run on a synthetic random
graph — 7,700 unnamed "units" wired by rng.randrange(), borrowing
only hemibrain degree statistics. No real neurons, no real
synapses, no cell identities (the user spotted it in the UI).

This module replaces that graph with four REAL subcircuits of the
FlyWire v783 connectome that already ships with the repo:

  DB-1 Market Pattern Analyzer   ALPN (685) + Kenyon cells (5,177)
                                 + MBON (96) — the mushroom-body
                                 pattern pathway, real synapses
  DB-2 User Experience Predictor LH (556) + LHLN (514) + MBON (96)
                                 + DAN (331) — the valence pathway
  DB-3 Market State Observer     CX (2,875) + DN (1,299) — the
                                 central-complex state pathway
  DB-4 Auto-Learning Controller  DAN (331) + DN (1,299) + MBON (96)
                                 — the teaching/output pathway

Every neuron is a real FlyWire root_id with a real cell class;
every edge is a real measured synapse from the v783 adjacency
(`W`, signed by presynaptic neurotransmitter, row-normalised here
for stable propagation). Each instance keeps its own activation
state per (asset, timeframe) and votes from its own specialty
readout — four different circuits, four different opinions.

If the FlyWire circuit is absent, BrainSystem falls back to the
legacy synthetic graph AND says so (never quoting real counts
for a synthetic graph again).
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
from scipy import sparse

log = logging.getLogger("real_brains")

APPROACH_NT = ("acetylcholine", "gaba")
_STATE_DIR = {"accumulation": 1, "markup": 1, "distribution": -1,
              "markdown": -1, "ranging": 0}


def _blocks(n: int, k: int) -> list[np.ndarray]:
    """Split range(n) into k contiguous index blocks."""
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


class RealCircuit:
    """One brain instance on a real induced subgraph."""

    def __init__(self, bid: str, role: str, idx: np.ndarray,
                 A: sparse.csr_matrix, groups: dict[str, np.ndarray]):
        self.id = bid
        self.role = role
        self.idx = idx                      # global FlyWire indices
        self.A = A                          # local signed adjacency
        self.groups = groups                # name -> local indices
        self.states: dict[tuple[str, str], np.ndarray] = {}
        self.n = len(idx)
        self.edges = int(A.nnz)

    def state_for(self, symbol: str, tf: str) -> np.ndarray:
        key = (symbol, tf)
        st = self.states.get(key)
        if st is None:
            st = np.zeros(self.n, dtype=np.float64)
            self.states[key] = st
        return st

    def sweep(self, x: np.ndarray, inp: np.ndarray,
              steps: int = 2) -> np.ndarray:
        for _ in range(max(1, steps)):
            x = np.tanh(self.A @ x + inp)
        return x

    def act(self, x: np.ndarray, name: str) -> float:
        g = self.groups.get(name)
        if g is None or len(g) == 0:
            return 0.0
        return float(np.mean(x[g]))


class RealBrains:
    """Builds + runs the four real subcircuits from a ready FlyWire."""

    def __init__(self, fw):
        self.fw = fw
        self.circuits: dict[str, RealCircuit] = {}
        self.built_ms = 0.0
        self._build()

    # ------------------------------------------------------------ build
    def _sub(self, idx: np.ndarray) -> sparse.csr_matrix:
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        W = self.fw.W
        A = W[idx][:, idx].tocsr().astype(np.float64)
        # row-normalise: a neuron's total incoming drive is bounded,
        # so propagation is a stable tanh recurrence (real synapse
        # signs and relative weights preserved)
        rows = np.asarray(np.abs(A).sum(axis=1)).ravel()
        scale = np.where(rows > 1.0, 1.0 / np.maximum(rows, 1e-9), 1.0)
        return (sparse.diags(scale) @ A).tocsr()

    def _mk(self, bid, role, idx, group_globals):
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        pos = {int(g): i for i, g in enumerate(idx)}
        groups = {}
        for name, gidx in group_globals.items():
            groups[name] = np.array([pos[int(g)] for g in gidx
                                     if int(g) in pos], dtype=np.int64)
        c = RealCircuit(bid, role, idx, self._sub(idx), groups)
        self.circuits[bid] = c
        return c

    def _build(self):
        t0 = time.perf_counter()
        fw = self.fw
        cls = fw.cls
        roots_idx = np.arange(len(fw.roots))
        cx = roots_idx[cls == "CX"]
        lhln = roots_idx[cls == "LHLN"]
        dn = roots_idx[fw.dn_mask] if fw.dn_mask is not None else \
            np.array([], dtype=np.int64)
        alpn, kc, mbon = fw.alpn, fw.kc, fw.mbon
        dan, lh = fw.dan, fw.lh
        nt = {int(g): str(n) for g, n in zip(mbon, fw.mbon_nt)}
        app = np.array([g for g in mbon if nt.get(int(g)) in APPROACH_NT])
        avo = np.array([g for g in mbon if nt.get(int(g)) == "glutamate"])

        self._mk("DB-1", "Market Pattern Analyzer",
                 np.concatenate([alpn, kc, mbon]),
                 {"input": alpn, "kc": kc, "mbon": mbon,
                  "approach": app, "avoid": avo})
        self._mk("DB-2", "User Experience Predictor",
                 np.concatenate([lh, lhln, dan, mbon]),
                 {"input_pos": lh[: len(lh) // 2],
                  "input_neg": lh[len(lh) // 2:],
                  "lh_pos": lh[: len(lh) // 2],
                  "lh_neg": lh[len(lh) // 2:],
                  "dan": dan, "mbon": mbon})
        self._mk("DB-3", "Market State Observer",
                 np.concatenate([cx, dn]),
                 {"cx": cx, "dn": dn})
        self._mk("DB-4", "Auto-Learning Controller",
                 np.concatenate([dan, dn, mbon]),
                 {"dan": dan, "dn": dn, "mbon": mbon,
                  "approach": app, "avoid": avo})
        self.built_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("real brain circuits built in %.0f ms: %s",
                 self.built_ms,
                 {k: (c.n, c.edges) for k, c in self.circuits.items()})

    # ------------------------------------------------------------ inputs
    def _input_vector(self, c: RealCircuit, thing2: dict) -> np.ndarray:
        inp = np.zeros(c.n)
        if c.id == "DB-1":
            rets = thing2.get("norm_returns20") or []
            blocks = _blocks(len(c.groups["input"]), 20)
            for i, b in enumerate(blocks):
                if len(b):
                    inp[b] = float(rets[i]) if i < len(rets) else 0.0
        elif c.id == "DB-2":
            tp = float(thing2.get("tp_dist") or 0.0)
            sl = float(thing2.get("sl_dist") or 0.0)
            den = tp + sl
            pos = tp / den if den > 0 else 0.5
            inp[c.groups["input_pos"]] = (pos - 0.5) * 3.0
            inp[c.groups["input_neg"]] = ((1.0 - pos) - 0.5) * 3.0
        elif c.id == "DB-3":
            cci = thing2.get("cc_inputs") or []
            blocks = _blocks(len(c.groups["cx"]), 5)
            for i, b in enumerate(blocks):
                if len(b):
                    inp[b] = (float(cci[i]) if i < len(cci) else 0.0) * 2.0
        elif c.id == "DB-4":
            perf = thing2.get("formula_perf") or {}
            vals = [float(v) for v in perf.values()]
            blocks = _blocks(len(c.groups["dan"]),
                             max(1, len(vals)))
            for i, b in enumerate(blocks):
                if len(b) and i < len(vals):
                    inp[b] = vals[i] * 2.0
        nv = float(thing2.get("news_value") or 0.0)
        if nv and "mbon" in c.groups:
            inp[c.groups["mbon"]] += max(-1.0, min(1.0, nv)) * 0.5
        return inp

    # ------------------------------------------------------------ evaluate
    def evaluate_instance(self, bid: str, symbol: str, tf: str,
                          thing2: dict, advance: bool = True) -> dict:
        c = self.circuits[bid]
        x0 = c.state_for(symbol, tf)
        x = c.sweep(x0.copy(), self._input_vector(c, thing2),
                    steps=2)
        if advance:
            c.states[(symbol, tf)] = x

        if bid == "DB-1":
            app = float(np.sum(x[c.groups["approach"]])) \
                if len(c.groups["approach"]) else 0.0
            avo = float(np.sum(x[c.groups["avoid"]])) \
                if len(c.groups["avoid"]) else 0.0
            signal = math.tanh(2.0 * (app - avo) / (abs(app) + abs(avo)
                                                    + 1e-9))
            mb = c.groups["mbon"]
            extra = {"pattern": signal,
                     "experience": 0.5,
                     "state_dist": None,
                     # legacy B1 valence hook consumes 12 pattern
                     # readouts in [0,1] — real MBON activities mapped
                     "pattern_act": [round(float((v + 1.0) / 2.0), 5)
                                     for v in x[mb][:12]]}
        elif bid == "DB-2":
            pos = c.act(x, "lh_pos")
            neg = c.act(x, "lh_neg")
            exp = (pos + 1.0) / ((pos + 1.0) + (neg + 1.0) + 1e-9)
            signal = (exp - 0.5) * 2.0
            extra = {"pattern": 0.0, "experience": round(exp, 5),
                     "state_dist": None}
        elif bid == "DB-3":
            from .config import BRAIN_MARKET_STATES
            blocks = _blocks(len(c.groups["cx"]), 5)
            acts = [float(np.mean(x[b])) if len(b) else 0.0
                    for b in blocks]
            mags = [abs(a) + 0.01 for a in acts]
            tot = sum(mags) or 1.0
            dist = [m / tot for m in mags]
            dom = int(np.argmax(dist))
            signal = _STATE_DIR.get(BRAIN_MARKET_STATES[dom], 0) \
                * dist[dom]
            extra = {"pattern": 0.0, "experience": 0.5,
                     "state_dist": {BRAIN_MARKET_STATES[i]:
                                    round(dist[i], 4) for i in range(5)},
                     "dominant": BRAIN_MARKET_STATES[dom],
                     "p_dominant": round(dist[dom], 4)}
        else:  # DB-4
            from .config import WEIGHT_IDS
            dn = c.groups["dn"]
            blocks = _blocks(len(dn), len(WEIGHT_IDS))
            base = float(np.mean(x[dn])) if len(dn) else 0.0
            wadj = {}
            for i, wid in enumerate(WEIGHT_IDS):
                b = blocks[i]
                wadj[wid] = round((float(np.mean(x[b])) - base)
                                  if len(b) else 0.0, 5)
            signal = math.tanh(4.0 * (sum(wadj.values())
                                      / max(1, len(wadj))))
            extra = {"pattern": 0.0, "experience": 0.5,
                     "state_dist": None, "weight_adj": wadj}

        hormones = {
            "dopamine": round(0.5 + 0.5 * c.act(x, "dan"), 4)
            if "dan" in c.groups else 0.5,
            "serotonin": round(0.5 + 0.5 * c.act(x, "mbon"), 4)
            if "mbon" in c.groups else 0.5,
            "octopamine": round(0.5 + 0.5 * c.act(x, "lh_pos"), 4)
            if "lh_pos" in c.groups else 0.5,
            "tyramine": round(0.5 + 0.5 * c.act(x, "dn"), 4)
            if "dn" in c.groups else 0.5,
        }
        extra.setdefault("pattern_act", [])
        return {"id": bid, "role": c.role,
                "signal": round(max(-1.0, min(1.0, float(signal))), 5),
                "h_signal": round(max(-1.0, min(1.0, float(signal))), 5),
                "hormones": hormones,
                "neurons": c.n, "synapses": c.edges,
                "backend": "flywire-v783-real",
                **extra}

    def summary(self) -> dict:
        return {"backend": "flywire-v783-real",
                "built_ms": self.built_ms,
                "circuits": {k: {"neurons": c.n, "synapses": c.edges,
                                 "role": c.role}
                             for k, c in self.circuits.items()},
                "total_neurons": int(sum(c.n for c in
                                         self.circuits.values())),
                "total_synapses": int(sum(c.edges for c in
                                          self.circuits.values()))}
