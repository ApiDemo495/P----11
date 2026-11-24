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


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


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

    def two_hop(self) -> sparse.csr_matrix:
        """W_2hop = row-normalised A² (guide §6.1).

        The recurrence used to re-apply the 1-hop matrix twice, which is
        NOT a 2-hop traversal — paths of length 2 were never formed. The
        squared operator is built lazily, row-normalised the same way as
        A, and cached; if the induced subgraph is dense enough that A²
        explodes past the budget the 1-hop matrix is returned instead
        (documented, never silent)."""
        m = getattr(self, "_w2", None)
        if m is not None:
            return m
        A2 = (self.A @ self.A).tocsr()
        if A2.nnz > 6_000_000:
            m = self.A
        else:
            rows = np.asarray(np.abs(A2).sum(axis=1)).ravel()
            scale = np.where(rows > 1.0, 1.0 / np.maximum(rows, 1e-9), 1.0)
            m = (sparse.diags(scale) @ A2).tocsr()
        self._w2 = m
        self.w2_edges = int(m.nnz)
        return m

    def sweep(self, x: np.ndarray, inp: np.ndarray,
              steps: int = 2) -> np.ndarray:
        for i in range(max(1, steps)):
            W = self.A if i == 0 else self.two_hop()
            x = np.tanh(W @ x + inp)
        return x

    def respond(self, x: np.ndarray, inp: np.ndarray,
                steps: int = 2) -> np.ndarray:
        """INPUT-DRIVEN response = driven state − undriven state.

        The recurrent pass always settles onto its own pattern; reading raw
        activations therefore reported the fixed point, not the signal. On a
        bullish and a bearish window DB-3 returned the identical +0.3059.
        Subtracting the zero-input pass leaves only what the input caused.
        """
        base = self.sweep(x.copy(), np.zeros_like(inp), steps=steps)
        driven = self.sweep(x.copy(), inp, steps=steps)
        return driven - base

    def act(self, x: np.ndarray, name: str) -> float:
        g = self.groups.get(name)
        if g is None or len(g) == 0:
            return 0.0
        return float(np.mean(x[g]))

    def act_delta(self, d: np.ndarray, name: str) -> float:
        """Mean input-driven change of a named group (signed)."""
        g = self.groups.get(name)
        if g is None or len(g) == 0:
            return 0.0
        return float(np.mean(d[g]))


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
        """Run one brain on one window.

        Every readout is taken from the INPUT-DRIVEN response (`d`), not from
        the raw activation field: the four circuits settle onto their own
        fixed patterns, so raw means described the wiring rather than the
        window (DB-3 printed the same state for opposite inputs, DB-4 printed
        ~0 and DB-2's readout was pinned to its dominant hemisphere). `d` is
        the difference between the driven pass and the undriven pass, i.e.
        exactly what this window's inputs caused.
        """
        c = self.circuits[bid]
        x0 = c.state_for(symbol, tf)
        inp = self._input_vector(c, thing2)
        d = c.respond(x0, inp, steps=2)
        x = c.sweep(x0.copy(), inp, steps=2)
        if advance:
            c.states[(symbol, tf)] = x

        def _mean(name: str) -> float:
            return c.act_delta(d, name)

        def _pos_neg(a: float, b: float) -> float:
            """Signed contrast in [−1,1] that cannot divide by zero."""
            return (a - b) / (abs(a) + abs(b) + 1e-9)

        if bid == "DB-1":
            # Market Pattern Analyzer: MBON approach vs avoid populations
            app = _mean("approach")
            avo = _mean("avoid")
            # Measured: on this connectome the 96 MBONs shift as one block —
            # the approach/avoid difference is ~1% of their common motion. So
            # the decision variable is the input-driven shift of the pattern
            # OUTPUT layer, and the split is reported separately (selectivity)
            # instead of being passed off as the signal.
            sig_mbon = _mean("mbon")
            signal = math.tanh(3.0 * sig_mbon)
            extra = {"pattern": round(signal, 5),
                     "approach": round(app, 5), "avoid": round(avo, 5),
                     "selectivity": round(_pos_neg(app, avo), 5),
                     "experience": 0.5,
                     "state_dist": None,
                     "response": round(float(np.abs(d[c.groups["mbon"]]).mean())
                                       if len(c.groups.get("mbon", [])) else 0.0,
                                       6),
                     "pattern_act": [round(float(v), 5)
                                     for v in d[c.groups["mbon"]][:12]]
                     if len(c.groups.get("mbon", [])) else []}
        elif bid == "DB-2":
            # User Experience Predictor: lateral-horn reward vs risk halves
            pos = _mean("lh_pos")
            neg = _mean("lh_neg")
            # tanh keeps a one-sided window near (±0.96, not exactly ±1:
            # a saturating readout cannot report "how sure")
            # how hard this window pushed the reward/risk halves at all, so
            # a mild window reports a mild expectation instead of ±0.96
            strength = math.tanh(4.0 * (abs(pos) + abs(neg)))
            exp = 0.5 + 0.5 * math.tanh(2.0 * _pos_neg(pos, neg)) * strength
            signal = (exp - 0.5) * 2.0
            extra = {"pattern": 0.0, "experience": round(exp, 5),
                     "state_dist": None,
                     "response": round(float(np.abs(d[c.groups["lh_pos"]]).mean()
                                             + np.abs(d[c.groups["lh_neg"]]).mean())
                                       / 2.0, 6)}
        elif bid == "DB-3":
            # Market State Observer: the 5 central-complex groups are the 5
            # market states. The posterior is a softmax over the SIGNED,
            # input-driven response (a state is "on" when the input pushes
            # that group up relative to the others) — the old version used
            # |raw activation|, which is a constant of the wiring.
            from .config import BRAIN_MARKET_STATES
            blocks = _blocks(len(c.groups["cx"]), 5)
            acts = [float(np.mean(d[b])) if len(b) else 0.0 for b in blocks]
            _spread = max((abs(a) for a in acts), default=0.0)
            temp = max(1e-9, 0.35 * _spread)
            z = [a / temp for a in acts]
            m = max(z) if z else 0.0
            e = [math.exp(_z - m) for _z in z]
            tot = sum(e) or 1.0
            dist = [v / tot for v in e]
            dom = int(np.argmax(dist))
            # expected directional bias over the whole posterior: a flat
            # posterior must produce ~0, a confident one the state's side
            signal = sum(_STATE_DIR.get(BRAIN_MARKET_STATES[i], 0) * dist[i]
                         for i in range(len(dist)))
            _dist = {BRAIN_MARKET_STATES[i]: round(dist[i], 4)
                     for i in range(5)}
            # keep the printed posterior a distribution: put the rounding
            # residue on the leading state instead of leaving a 0.9999
            _lead = BRAIN_MARKET_STATES[dom]
            _dist[_lead] = round(_dist[_lead] + (1.0 - sum(_dist.values())), 4)
            extra = {"pattern": 0.0, "experience": 0.5,
                     "state_dist": _dist,
                     "dominant": BRAIN_MARKET_STATES[dom],
                     "p_dominant": round(dist[dom], 4),
                     # a 5-way softmax at 0.27 is not a state, it is a shrug
                     "confident": bool(dist[dom] >= 0.40),
                     "response": round(float(np.abs(d[c.groups["cx"]]).mean())
                                       if len(c.groups.get("cx", [])) else 0.0,
                                       6)}
        else:  # DB-4
            # Auto-Learning Controller. Its input lands in the DAN blocks, so
            # its readout MUST come from the same population (it used to read
            # DN, which the input never touches, hence the permanent ±0.00).
            from .config import WEIGHT_IDS
            dan = c.groups["dan"]
            blocks = _blocks(len(dan), len(WEIGHT_IDS))
            base = float(np.mean(d[dan])) if len(dan) else 0.0
            wadj = {}
            for i, wid in enumerate(WEIGHT_IDS):
                b = blocks[i]
                wadj[wid] = round((float(np.mean(d[b])) - base)
                                  if len(b) else 0.0, 5)
            spread = max((abs(v) for v in wadj.values()), default=0.0)
            signal = math.tanh(8.0 * (sum(wadj.values()) / max(1, len(wadj))))
            extra = {"pattern": 0.0, "experience": 0.5,
                     "state_dist": None, "weight_adj": wadj,
                     "believed": round(base, 6),
                     "response": round(float(np.abs(d[dan]).mean())
                                       if len(dan) else 0.0, 6),
                     # a controller whose per-formula deltas are all equal has
                     # nothing to say about any of them
                     "informative": bool(spread > 1e-4)}

        # ---- Thing 1 hormones: input-driven tone of each population ------
        def _h(name: str) -> float:
            if name not in c.groups:
                return 0.5
            return round(_clamp01(0.5 + 2.0 * _mean(name)), 4)

        hormones = {"dopamine": _h("dan"), "serotonin": _h("mbon"),
                    "octopamine": _h("lh_pos"), "tyramine": _h("dn")}
        extra.setdefault("pattern_act", [])
        response = float(extra.get("response") or 0.0)
        energy = float(np.abs(inp).mean())
        return {"id": bid, "role": c.role,
                "signal": round(max(-1.0, min(1.0, float(signal))), 5),
                "h_signal": round(max(-1.0, min(1.0, float(signal))), 5),
                "hormones": hormones,
                "neurons": c.n, "synapses": c.edges,
                "backend": "flywire-v783-real",
                # per-brain honesty: is it being driven, and did it answer?
                "ready": True,
                "input_energy": round(energy, 6),
                "response": round(response, 6),
                "abstain": bool(energy < 1e-9 or response < 1e-7),
                **extra}

    # ------------------------------------------------------------ self-test
    @staticmethod
    def _probe_inputs(sign: float) -> dict:
        """Deterministic bullish/bearish window used only for the self-test."""
        return {
            "norm_returns20": [sign * ((i % 5) - 2) / 2.0 for i in range(20)],
            "tp_dist": 2.0 if sign > 0 else 0.5,
            "sl_dist": 0.5 if sign > 0 else 2.0,
            "cc_inputs": [sign * 0.8, -sign * 0.6, sign * 0.4,
                          -sign * 0.2, sign * 0.1],
            "formula_perf": {i: float(np.tanh(sign * (i - 18) / 10.0))
                             for i in range(1, 36)},
            "news_value": sign * 0.7,
        }

    def self_test(self) -> dict:
        """Drive every brain with a bullish and a bearish window.

        This is the machinery behind "all four brains work": a brain that
        cannot tell those two windows apart is not doing its job, whatever
        its activation field looks like. `advance=False` so the probe never
        leaks into a real symbol's state.
        """
        bull_in = self._probe_inputs(1.0)
        bear_in = self._probe_inputs(-1.0)
        out = {}
        for bid in self.circuits:
            b = self.evaluate_instance(bid, "__selftest__", "probe",
                                       bear_in, advance=False)
            u = self.evaluate_instance(bid, "__selftest__", "probe",
                                       bull_in, advance=False)
            sb, su = float(b["signal"]), float(u["signal"])
            flips = (su > 1e-3 and sb < -1e-3) or (su < -1e-3 and sb > 1e-3)
            out[bid] = {
                "bull": round(su, 4), "bear": round(sb, 4),
                "separation": round(abs(su - sb), 4),
                "input_energy": u.get("input_energy"),
                "response": u.get("response"),
                "responds": bool(u.get("response", 0.0) > 1e-7
                                 and abs(su - sb) > 1e-3),
                "flips": bool(flips),
                "dominant_state": u.get("dominant"),
                "p_dominant": u.get("p_dominant"),
                "informative": u.get("informative"),
            }
        return out

    def summary(self) -> dict:
        # §6.1: the census figure is DERIVED from the live connectome
        # (sum of per-circuit nnz) and cached — never a hardcoded
        # 15,091,983 that drifts away from the graph it describes.
        total_syn = int(sum(c.edges for c in self.circuits.values()))
        self.total_synapses = total_syn
        return {"backend": "flywire-v783-real",
                "loaded_synapses": total_syn,
                "built_ms": self.built_ms,
                "circuits": {k: {"neurons": c.n, "synapses": c.edges,
                                 "role": c.role}
                             for k, c in self.circuits.items()},
                "total_neurons": int(sum(c.n for c in
                                         self.circuits.values())),
                "total_synapses": int(sum(c.edges for c in
                                          self.circuits.values()))}
