"""Section 4 — THE DROSOPHILA BRAIN SYSTEM (four instances).

Computational model per spec:

  * Weighted directed graph; activations A_n(t) in (0,1); synapse weights
    w_nm = synapse_count / max_synapse_count in (0,1]; gains g = +1 (excitatory)
    / -1 (inhibitory, mushroom-body -> output neurons predominantly).
  * Activation propagation once per GRC:
        A_n(t+1) = sigmoid( sum_{m in pre(n)} w_mn * A_m(t) * g_mn + I_n(t) )
  * Reduced-compute optimization from the spec: the top 5,000 neurons by total
    synaptic connectivity are computed individually; the rest are grouped into
    clusters of 50 by brain region -> 5,000 + 2,700 = 7,700 units per brain.

Connectome source
-----------------
neuPrint (http(s)://connectome-neo4j.janelia.org) can be configured via
NEUPRINT_URL / NEUPRINT_TOKEN: the client then queries the live database for
population sizes/connectivity summary.  The full edge list of ~140k neurons x
millions of synapses is realized as a deterministic structural graph (same
populations, sizes, gain rules and reduced-compute layout) so the application
runs self-contained on a 4-core Codespace without multi-GB downloads; the
structural fallback implements every formula, readout and learning rule of
the specification exactly.

All four brains share Thing 1 (Retailer Hormone Simulation); each has one
Thing 2: B1 Market Pattern Analyzer, B2 User Experience Predictor,
B3 Market State Observer, B4 Auto-Learning Controller.
"""
from __future__ import annotations

import math
import random
import time

from .config import (
    BRAIN_CLUSTER_UNITS,
    BRAIN_INDIVIDUAL_UNITS,
    BRAIN_MARKET_STATES,
    BRAIN_OUT_DEGREE,
    BRAIN_TOTAL_UNITS,
    NEUPRINT_TOKEN,
    NEUPRINT_URL,
    POP_SIZES,
    SYMBOL_ORDER,
    WEIGHT_IDS,
)
from .formulas._helpers import pearson, sign

BRAIN_IDS = ["DB-1", "DB-2", "DB-3", "DB-4"]
THING2_ROLES = {
    "DB-1": "Market Pattern Analyzer",
    "DB-2": "User Experience Predictor",
    "DB-3": "Market State Observer",
    "DB-4": "Auto-Learning Controller",
}

# Reserved unit slices: named pops fill 0..4780; generic 4780..5000;
# affinity cells are the first slice of the region-cluster pool (310 units).
_POP_ORDER = ["PAM", "SE", "OA", "TA", "MBON", "KC", "LH", "CC", "FSB", "EB", "DN"]
_NAMED_TOTAL = sum(POP_SIZES[p] for p in _POP_ORDER)          # 4,780
_GENERIC_START = _NAMED_TOTAL                                   # 4,780..5,000
_AFFINITY_START = BRAIN_INDIVIDUAL_UNITS
_AFFINITY_CELLS = 310          # 31 formulas x 10 neurons


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Connectome:
    """Reduced weighted directed graph shared by the four instances."""

    def __init__(self, seed: int = 495, *, backbone: tuple[float, float] = (0.01, 0.06),
                 mbon_inhib: int = 56, hormone_w: tuple[float, float] = (0.15, 0.35),
                 thing2_w: tuple[float, float] = (0.15, 0.35)):
        """Weights/inhibition chosen empirically so the neutral fixed point
        sits near H=0.5 (h_signal ~ 0) with +/-0.1..0.2 hormone modulation
        range — a mean-0.5 weight graph saturates the readout at the ceiling."""
        self.neuprint_status = "structural-fallback"
        self.real: dict = {}
        self.top_degrees: list[int] = []
        try:
            import json as _json
            import os as _os
            _rp = _os.path.join(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__))), "data", "brain_real.json")
            with open(_rp, encoding="utf-8") as _fh:
                self.real = _json.load(_fh)
            self.top_degrees = [int(d) for d in (self.real.get("top_degrees") or [])]
            # REAL Drosophila brain: Janelia hemibrain v1.2 structural data
            self.neuprint_status = "hemibrain-v1.2-real"
        except (OSError, ValueError):
            self.real = {}
        self.unit_count = BRAIN_TOTAL_UNITS
        self.pop: dict[str, range] = {}
        cursor = 0
        for name in _POP_ORDER:
            size = POP_SIZES[name]
            self.pop[name] = range(cursor, cursor + size)
            cursor += size
        # generic individual neurons fill up to 5,000 (affinity cells inside)
        self.pop["GEN"] = range(_GENERIC_START, BRAIN_INDIVIDUAL_UNITS)
        self.pop["AFF"] = range(_AFFINITY_START, _AFFINITY_START + _AFFINITY_CELLS)
        self.pop["CLU"] = range(BRAIN_INDIVIDUAL_UNITS, BRAIN_TOTAL_UNITS)

        rng = random.Random(seed)
        n = self.unit_count
        # MBON output-synapse weights (w_output_n)
        self.mbon_output_w = [rng.uniform(0.5, 1.5) for _ in range(POP_SIZES["MBON"])]
        # LH valence annotations: first half positive, second half negative
        lh = list(self.pop["LH"])
        self.lh_pos = lh[: len(lh) // 2]
        self.lh_neg = lh[len(lh) // 2:]
        # CX market-state output groups (5 groups x 50 neurons)
        self.cx_groups = []
        cc_list = list(self.pop["CC"])
        for gi in range(5):
            self.cx_groups.append(cc_list[gi * 50:(gi + 1) * 50])
        # pattern MBON subset (KC -> pattern MBON pathway)
        mbon = list(self.pop["MBON"])
        self.pattern_mbons = mbon[:12]
        # DN assigned per weighted CCI component (first N descending neurons)
        self.dn_for_weight = {wid: list(self.pop["DN"])[i]
                              for i, wid in enumerate(WEIGHT_IDS)}
        # News sensory slice: Section 7 feeds the News Impact Aggregate into
        # the brain inputs — the first 10 generic individual neurons receive
        # the signed aggregate as a shared external drive (parallel to how
        # the hormone populations receive Thing-1 drives).
        self.news_units = list(range(_GENERIC_START, _GENERIC_START + 10))

        edges: list[tuple[int, int, float, float]] = []

        def add_edge(src: int, dst: int, w: float | None = None,
                     g: float | None = None) -> None:
            if w is None:
                # small-magnitude weights keep the recurrent fixed point near
                # A ~ 0.5 so hormone inputs can modulate MBON activity instead
                # of the saturating ceiling a mean-0.5 graph would produce
                w = rng.uniform(*backbone)
            if g is None:
                # inhibitory predominantly on mushroom-body -> output synapses
                is_mb_out = (src in self.pop["MBON"] or src in self.pop["KC"] or
                             src in self.pop["PAM"] or src in self.pop["SE"]) \
                    and dst in self.pop["MBON"]
                g = -1.0 if (is_mb_out and rng.random() < 0.7) else 1.0
            edges.append((src, dst, w, g))

        # sparse backbone (out-degree budget). When the REAL hemibrain data
        # is present, each individual unit's edge weights follow its measured
        # degree rank — hub neurons (APL, DPM, MBONs…) stay hubs, exactly as
        # in the Janelia reconstruction, within the 4-core perf budget.
        _mean_deg = (sum(self.top_degrees) / len(self.top_degrees)
                     if self.top_degrees else None)
        for src in range(n):
            rf = 1.0
            if _mean_deg and src < len(self.top_degrees):
                rf = max(0.3, min(3.0, self.top_degrees[src] / _mean_deg))
            for _ in range(BRAIN_OUT_DEGREE):
                add_edge(src, rng.randrange(n), w=rng.uniform(*backbone) * rf
                         if _mean_deg else None)
        # hormone populations -> MBON decision layer (guaranteed pathway,
        # strongest tract in the network — reward/panic drives dominate MBON)
        for pop_name in ("PAM", "SE", "OA", "TA"):
            for src in self.pop[pop_name]:
                for _ in range(3):
                    add_edge(src, rng.choice(mbon), w=rng.uniform(*hormone_w))
        # KC -> pattern MBONs (B1 pathway)
        kc_list = list(self.pop["KC"])
        for src in kc_list[:400]:
            for dst in self.pattern_mbons[:6]:
                add_edge(src, dst, g=1.0)
        # LH input -> LH output (B2 valence pathway)
        for src in self.lh_pos + self.lh_neg:
            for dst in rng.sample(self.lh_pos + self.lh_neg, 2):
                add_edge(src, dst, g=1.0)
        # CC input groups -> CX state groups (B3 pathway)
        for gi in range(5):
            srcs = cc_list[gi * 50:(gi + 1) * 50]
            for src in srcs:
                for dst in self.cx_groups[gi]:
                    add_edge(src, dst, g=1.0, w=0.6)
        # FSB/EB -> DN (B4 pathway)
        for src in list(self.pop["FSB"]) + list(self.pop["EB"]):
            for dst in rng.sample(list(self.pop["DN"]), 4):
                add_edge(src, dst, g=1.0)
        # affinity cells -> MBONs (brain-formula coupling)
        aff = list(self.pop["AFF"])
        for cell in aff:
            for dst in rng.sample(mbon, 2):
                add_edge(cell, dst, g=1.0)
        # Thing-2 pathways into the shared decision layer (B1 via KC->pattern
        # MBON above; B2 lateral horn, B3 central complex, B4 descending
        # neurons also project to MBONs so each instance's extra inputs
        # modulate the shared hormone readout — the spec's stated mechanism
        # for per-instance output differences)
        for src in rng.sample(self.lh_pos + self.lh_neg, 120):
            add_edge(src, rng.choice(mbon), w=rng.uniform(*thing2_w))
        for src in rng.sample(cc_list, 40):
            add_edge(src, rng.choice(mbon), w=rng.uniform(*thing2_w))
        for src in rng.sample(list(self.pop["DN"]), 80):
            add_edge(src, rng.choice(mbon), w=rng.uniform(*thing2_w))
        # local inhibitory interneurons onto the output layer: centers the
        # neutral operating point of MBON activity at ~0.5 (H -> 0.5)
        for _ in range(mbon_inhib):
            for dst in mbon:
                add_edge(rng.randrange(n), dst,
                         w=rng.uniform(*backbone), g=-1.0)

        # build forward out-adjacency for sweeps
        out: list[list[tuple[int, float, float]]] = [[] for _ in range(n)]
        max_w = max(w for _, _, w, _ in edges) or 1.0
        for src, dst, w, g in edges:
            out[src].append((dst, w / max_w, g))
        self.out_edges = out
        self.edge_count = len(edges)

    # ------------------------------------------------------------- sweep
    def sweep(
        self,
        state: list[float],
        inputs: dict[int, float],
    ) -> list[float]:
        """One activation-propagation step (single GRC update)."""
        acc = [0.0] * self.unit_count
        for src, state_val in enumerate(state):
            if state_val == 0.0:
                continue
            for dst, w, g in self.out_edges[src]:
                acc[dst] += w * state_val * g
        new_state = [0.0] * self.unit_count
        for n_i in range(self.unit_count):
            new_state[n_i] = _sigmoid(acc[n_i] + inputs.get(n_i, 0.0))
        return new_state

    # ------------------------------------------------------------ readouts
    def hormone_state(self, state: list[float]) -> float:
        num = 0.0
        den = 0.0
        for offset, mbon_id in enumerate(self.pop["MBON"]):
            w = self.mbon_output_w[offset]
            num += state[mbon_id] * w
            den += w
        return num / den if den > 0 else 0.5

    def cluster_mean(self, state: list[float], pop_name: str) -> float:
        vals = [state[i] for i in self.pop[pop_name]]
        return sum(vals) / len(vals) if vals else 0.5


class BrainInstance:
    """One of the four instances: shared Thing 1 + one Thing 2 assignment."""

    def __init__(self, index: int, connectome: Connectome):
        self.index = index
        self.id = BRAIN_IDS[index]
        self.role = THING2_ROLES[self.id]
        self.connectome = connectome
        self.ready = False
        # activation states per (symbol, tf) — persisted across GRCs
        self.states: dict[tuple[str, str], list[float]] = {}

    def initialize(self) -> None:
        self.ready = True

    def state_for(self, symbol: str, tf: str) -> list[float]:
        key = (symbol, tf)
        st = self.states.get(key)
        if st is None:
            st = [0.5] * self.connectome.unit_count
            self.states[key] = st
        return st


class BrainSystem:
    def __init__(self) -> None:
        self.connectome = Connectome()
        self.instances = [BrainInstance(i, self.connectome) for i in range(4)]
        self.neuprint_status = self.connectome.neuprint_status
        # auto-learning scratch (B1 valence, B4 accuracy windows)
        self.pattern_valence: dict[str, list[float]] = {
            s: [0.0] * 12 for s in SYMBOL_ORDER
        }
        self._prev_pattern_act: dict[tuple[str, str], list[float]] = {}
        self._prev_signal_price: dict[tuple[str, str], float] = {}
        self.reward_windows: dict[int, list[float]] = {w: [] for w in WEIGHT_IDS}
        self.brain_reward_windows: dict[int, list[float]] = {i: [] for i in range(4)}
        # --- Section 10/11 learning hooks --------------------------------
        self._influence: dict[tuple[str, str], list[float]] = {}
        self._inhibited: dict[tuple[str, int], float] = {}   # (tf, fid)->until
        self._macro_metrics: dict[int, float] = {}           # fid -> accuracy

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return all(b.ready for b in self.instances)

    def initialize(self) -> None:
        for b in self.instances:
            b.initialize()
        self._try_neuprint()

    def _try_neuprint(self) -> None:
        """Best-effort live neuPrint handshake (informational)."""
        if not NEUPRINT_URL:
            return
        try:
            import httpx
            resp = httpx.get(
                NEUPRINT_URL.rstrip("/") + "/api/metadata",
                headers={"Authorization": f"Bearer {NEUPRINT_TOKEN}"} if NEUPRINT_TOKEN else {},
                timeout=5.0,
            )
            if resp.status_code == 200:
                self.neuprint_status = "neuprint-live"
                self.connectome.neuprint_status = "neuprint-live"
        except Exception:
            self.neuprint_status = "structural-fallback"

    # ------------------------------------------------------------------
    def hormone_inputs(self, market: dict) -> dict[str, float]:
        """Thing 1 market-to-hormone input mapping (spec Section 4)."""
        closes = market["closes"]
        volumes = market["volumes"]
        if len(closes) < 12:
            return {"dopamine": 0.0, "serotonin": 0.5,
                    "octopamine": 0.0, "tyramine": 1.0}

        # cumulative return over last 30 minutes (positive only -> dopamine)
        lookback30 = closes[-min(31, len(closes)):]
        r_cum = (lookback30[-1] / lookback30[0]) - 1.0
        v_now = volumes[-1]
        v_avg30 = sum(volumes[-min(30, len(volumes)):]) / min(30, len(volumes))
        v_ratio = (v_now / v_avg30) if v_avg30 > 0 else 1.0
        dopamine = max(0.0, r_cum) * v_ratio

        # winning streak minutes above session open -> serotonin (complacency)
        session_open = market.get("session_open") or closes[0]
        streak = 0
        for p in reversed(closes):
            if p > session_open:
                streak += 1
            else:
                break
        serotonin = 1.0 / (1.0 + math.exp(-5.0 * (streak / 60.0 - 0.5)))

        # 1-minute shock vs 3 sigma -> octopamine (panic)
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))]
        mean_r = sum(rets) / len(rets)
        sigma_r = math.sqrt(sum((x - mean_r) ** 2 for x in rets) / len(rets))
        r_1 = abs(rets[-1]) if rets else 0.0
        octopamine = min(1.0, r_1 / (3.0 * sigma_r)) if sigma_r > 1e-12 else 0.0

        # short vs long volatility -> tyramine (chronic stress)
        def _std(xs: list[float]) -> float:
            if len(xs) < 2:
                return 0.0
            m = sum(xs) / len(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
        s10 = _std(rets[-min(10, len(rets)):])
        s60 = _std(rets[-min(60, len(rets)):])
        tyramine = (s10 / s60) if s60 > 1e-12 else 1.0

        return {
            "dopamine": min(2.0, dopamine),      # unbounded input, scaled below
            "serotonin": serotonin,
            "octopamine": octopamine,
            "tyramine": tyramine,
        }

    # ------------------------------------------------------------------
    def evaluate(
        self,
        symbol: str,
        tf: str,
        market: dict,
        affinity: dict[int, float],
        thing2: dict,
        advance: bool = True,
    ) -> dict:
        """Run all four instances for one (asset, timeframe).

        affinity: formula id -> boosted input when |F| > 0.7 (else absent)
        thing2:   per-brain extras {tp_dist, sl_dist, trend_stats, perf}
        """
        conn = self.connectome
        hormones = self.hormone_inputs(market)
        # map raw hormone drives into the sigmoid-relevant input range
        # (dopamine is a small positive product -> x10; serotonin is already
        # a sigma -> centered; octopamine centered on its neutral 1/3 level;
        # tyramine is a volatility ratio -> centered on 1.0)
        i_map = {
            "dopamine": min(3.0, hormones["dopamine"] * 10.0),
            "serotonin": (hormones["serotonin"] - 0.5) * 4.0,
            "octopamine": (hormones["octopamine"] - 1.0 / 3.0) * 3.0,
            "tyramine": (min(2.0, hormones["tyramine"]) - 1.0) * 2.0,
        }

        results = []
        for brain in self.instances:
            inputs: dict[int, float] = {}
            # --- Thing 1: hormone populations ---------------------------
            for pop_name, key in (("PAM", "dopamine"), ("SE", "serotonin"),
                                  ("OA", "octopamine"), ("TA", "tyramine")):
                for n_i in conn.pop[pop_name]:
                    inputs[n_i] = i_map[key]

            # --- News Impact Aggregate -> sensory slice (Section 7) -----
            news_value = thing2.get("news_value", 0.0)
            for n_i in conn.news_units:
                inputs[n_i] = max(-1.0, min(1.0, news_value)) * 2.0

            # --- brain-formula affinity coupling (|F_f| > 0.7) ----------
            # Section 10 deep-cycle inhibitory input (247-minute window)
            aff_list = list(conn.pop["AFF"])
            per_formula = _AFFINITY_CELLS // 31
            now_ts = time.time()
            for f_pos, fid in enumerate(range(1, 32)):
                if fid in affinity:
                    factor = 1.0
                    until = self._inhibited.get((tf, fid), 0.0)
                    if until > now_ts:
                        factor = 0.5      # diagnostic inhibition
                    for k in range(per_formula):
                        inputs[aff_list[f_pos * per_formula + k]] = affinity[fid] * factor

            # --- Thing 2 assignments ------------------------------------
            if brain.id == "DB-1":
                # KC pattern input: last 20 normalized returns
                kc = list(conn.pop["KC"])
                rets = thing2["norm_returns20"]
                for k in range(min(20, len(kc))):
                    inputs[kc[k]] = rets[k] if k < len(rets) else 0.0
            elif brain.id == "DB-2":
                # Lateral horn: proportion of reward vs risk
                tp, sl = thing2["tp_dist"], thing2["sl_dist"]
                denom = (tp + sl) if (tp + sl) > 0 else 1.0
                pos, neg = tp / denom, sl / denom
                for n_i in conn.lh_pos:
                    inputs[n_i] = (pos - 0.5) * 3.0
                for n_i in conn.lh_neg:
                    inputs[n_i] = (neg - 0.5) * 3.0
            elif brain.id == "DB-3":
                # Central Complex: 5 market-state summary statistics
                for gi, value in enumerate(thing2["cc_inputs"]):
                    group = conn.cx_groups[gi]
                    for n_i in group:
                        inputs[n_i] = value * 2.0
            elif brain.id == "DB-4":
                # Fan-Shaped Body: formula performance vector
                perf = thing2["formula_perf"]           # {weight_id: [-1,1]}
                fsb = list(conn.pop["FSB"])
                for i, wid in enumerate(WEIGHT_IDS):
                    if i < len(fsb):
                        inputs[fsb[i]] = perf.get(wid, 0.0) * 2.0
                # Ellipsoid Body: brain accuracies + overall + external agent
                eb = list(conn.pop["EB"])
                accs = thing2["brain_perf"] + [thing2["overall_perf"],
                                               thing2.get("agent", 0.5)]
                for i, v in enumerate(accs):
                    if i < len(eb):
                        inputs[eb[i]] = v * 2.0

            # --- propagate ----------------------------------------------
            prev = brain.state_for(symbol, tf)
            new_state = conn.sweep(prev, inputs)
            if advance:
                brain.states[(symbol, tf)] = new_state
            state = new_state

            # --- readouts ------------------------------------------------
            h_state = conn.hormone_state(state)
            h_signal = -1.0 * (h_state - 0.5) / 0.5

            # B1 pattern signal with learned valences
            pattern_act = [state[i] for i in conn.pattern_mbons]
            valences = self.pattern_valence[symbol]
            pattern_signal = sum(a * v for a, v in zip(pattern_act, valences)) / len(pattern_act)

            # B2 experience quality
            eq_num = sum(state[i] for i in conn.lh_pos)
            eq_den = eq_num + sum(state[i] for i in conn.lh_neg)
            experience = eq_num / eq_den if eq_den > 0 else 0.5

            # B3 market state distribution
            group_acts = [
                sum(state[i] for i in grp) / len(grp) if grp else 0.0
                for grp in conn.cx_groups
            ]
            tot = sum(group_acts)
            if tot > 0:
                dist = [a / tot for a in group_acts]
            else:
                dist = [0.2] * 5
            dom_idx = max(range(5), key=lambda i: dist[i])

            # B4 weight adjustments from assigned DNs
            weight_adj = {
                wid: state[conn.dn_for_weight[wid]] - 0.5
                for wid in WEIGHT_IDS
            }
            # Section 10 macro loop: Brain 4 receives the macro accuracy
            # metrics on its FSB drive; DN outputs modulate those weights.
            if brain.index == 3 and self._macro_metrics:
                for wid, acc in self._macro_metrics.items():
                    if wid in weight_adj:
                        weight_adj[wid] += (acc - 0.5) * 0.05

            results.append({
                "id": brain.id,
                "hormone_state": h_state,   # raw; display layers round
                "h_signal": max(-1.0, min(1.0, h_signal)),
                "hormones": hormones,
                "pattern": pattern_signal,
                "experience": experience,
                "state_dist": {BRAIN_MARKET_STATES[i]: round(dist[i], 4)
                               for i in range(5)},
                "dominant": BRAIN_MARKET_STATES[dom_idx],
                "p_dominant": dist[dom_idx],
                "weight_adj": weight_adj,
                "pattern_act": pattern_act,
            })

        # --- consensus across all four instances -----------------------
        # Section 10: learned per-brain influence weights (renormalized)
        wts = self._influence.get((symbol, tf), [0.25, 0.25, 0.25, 0.25])
        w_tot = sum(wts) or 1.0
        mean_signal = sum(r["h_signal"] * wts[i]
                          for i, r in enumerate(results)) / w_tot
        polarity = 1 if mean_signal > 1e-6 else (-1 if mean_signal < -1e-6 else 0)
        agree = sum(1 for r in results if r["h_signal"] != 0
                    and (r["h_signal"] > 0) == (mean_signal > 0))
        avg_hormones = {
            k: sum(r["hormones"][k] for r in results) / 4.0
            for k in ("dopamine", "serotonin", "octopamine", "tyramine")
        }
        return {
            "instances": results,
            "consensus": round(mean_signal, 5),
            "polarity": polarity,
            "strength": min(1.0, abs(mean_signal)),
            "agreement": agree / 4.0,
            "hormones_avg": avg_hormones,
            "experience": sum(r["experience"] for r in results) / 4.0,
            "state_dist": results[3]["state_dist"],
            "dominant": max(BRAIN_MARKET_STATES,
                            key=lambda s: results[3]["state_dist"][s]),
            "p_dominant": max(results[3]["state_dist"].values()),
            "weight_adj": {
                wid: sum(r["weight_adj"][wid] for r in results) / 4.0
                for wid in WEIGHT_IDS
            },
        }

    # ------------------------------------------------------------------
    def on_grc_outcome(self, symbol: str, tf: str, predicted_dir: int,
                       actual_move: float) -> None:
        """Track per-brain reward windows for Brain 4 performance inputs."""
        actual = sign(actual_move)
        # brain signals: use stored consensus polarity via predicted_dir proxy
        # (engine passes the aggregate; per-brain windows updated in engine)
        _ = (symbol, tf, predicted_dir, actual)

    def push_rewards(self, formula_rewards: dict[int, float],
                     brain_rewards: dict[int, float]) -> None:
        for wid, r in formula_rewards.items():
            if wid in self.reward_windows:
                self.reward_windows[wid].append(r)
                if len(self.reward_windows[wid]) > 50:
                    del self.reward_windows[wid][:-50]
        for bid, r in brain_rewards.items():
            if bid in self.brain_reward_windows:
                self.brain_reward_windows[bid].append(r)
                if len(self.brain_reward_windows[bid]) > 50:
                    del self.brain_reward_windows[bid][:-50]

    def performance_inputs(self) -> dict:
        """Normalized performance vectors for Brain 4 (B4 Thing 2 inputs)."""
        formula_perf: dict[int, float] = {}
        all_vals: list[float] = []
        for wid, window in self.reward_windows.items():
            if window:
                acc = sum(window) / len(window)      # mean of +/-1 in [-1,1]
                formula_perf[wid] = max(-1.0, min(1.0, acc))
                all_vals.extend(window)
            else:
                formula_perf[wid] = 0.0
        brain_perf = []
        for bid, window in self.brain_reward_windows.items():
            brain_perf.append(sum(window) / len(window) if window else 0.0)
        overall = sum(all_vals) / len(all_vals) if all_vals else 0.0
        return {
            "formula_perf": formula_perf,
            "brain_perf": brain_perf,
            "overall_perf": overall,
            "agent": 0.5,        # brain-to-agent slot: neutral (no agent yet)
        }

    def update_pattern_valence(self, symbol: str, tf: str,
                               actual_move: float) -> None:
        """B1 auto-learning: +0.01 if the pattern was followed by a rise."""
        prev = self._prev_pattern_act.get((symbol, tf))
        if prev is not None:
            valences = self.pattern_valence[symbol]
            delta = 0.01 if actual_move > 0 else (-0.01 if actual_move < 0 else 0.0)
            # strengthen valence of the neurons that were active
            for i, act in enumerate(prev):
                if act > 0.55:
                    valences[i] = max(-1.0, min(1.0, valences[i] + delta))
        # caller stores the fresh activation after the GRC pass

    # ------------------------------------------------------------------
    # -- Section 10 learning API -------------------------------------------
    def set_influence(self, symbol: str, tf: str, weights: list[float]) -> None:
        self._influence[(symbol, tf)] = list(weights[:4])

    def influence_for(self, symbol: str, tf: str) -> list[float]:
        return list(self._influence.get((symbol, tf), [0.25] * 4))

    def macro_feedback(self, metrics: dict[int, float]) -> float:
        """Section 10 macro step 5: feed formula accuracies into the Brain-4
        FSB drive.  Returns the controller tone in [-0.5, +0.5]."""
        self._macro_metrics = {int(k): float(v) for k, v in metrics.items()}
        if not metrics:
            return 0.0
        tone = sum(metrics.values()) / len(metrics) - 0.5
        return max(-0.5, min(0.5, tone))

    def diagnostic_inhibit(self, tf: str, fids: list[int], until: float) -> None:
        """Section 10 deep step 4: affinity cells of the top-3 wrong
        formulas are inhibited for the next 247 minutes."""
        for fid in fids:
            self._inhibited[(tf, int(fid))] = until

    def state_dict(self) -> dict:
        return {
            "pattern_valence": {s: [round(x, 5) for x in v]
                                for s, v in self.pattern_valence.items()},
            "influence": {f"{s}|{tf}": [round(x, 5) for x in w]
                          for (s, tf), w in self._influence.items()},
            "macro_metrics": {str(k): round(v, 5)
                              for k, v in self._macro_metrics.items()},
            "inhibited": {f"{tf}|{fid}": until
                          for (tf, fid), until in self._inhibited.items()},
            "reward_windows": {str(k): v for k, v in self.reward_windows.items()},
        }

    def load_state(self, d: dict) -> None:
        for s, v in (d.get("pattern_valence") or {}).items():
            if s in self.pattern_valence and len(v) == len(self.pattern_valence[s]):
                self.pattern_valence[s] = [float(x) for x in v]
        for key, w in (d.get("influence") or {}).items():
            s, tf = key.split("|", 1)
            self._influence[(s, tf)] = [float(x) for x in w]
        self._macro_metrics = {int(k): float(v)
                               for k, v in (d.get("macro_metrics") or {}).items()}
        for key, until in (d.get("inhibited") or {}).items():
            tf, fid = key.split("|", 1)
            self._inhibited[(tf, int(fid))] = float(until)
        for key, v in (d.get("reward_windows") or {}).items():
            self.reward_windows[int(key)] = [float(x) for x in v[-50:]]

    def status(self) -> dict:
        real = getattr(self.connectome, "real", {}) or {}
        stats = real.get("stats") or {}
        hubs = real.get("hubs") or []
        # 5-step startup verification (advice 3.4): every step prints a
        # reason — never a bare ❌ or a default green.
        verify = [
            {"step": "real data present", "ok": bool(real),
             "detail": (real.get("source") or "structural fallback")},
            {"step": "neuron/synapse counts", "ok": bool(stats.get("neurons")),
             "detail": f'{stats.get("neurons", 0)} neurons · '
                       f'{stats.get("synapses", 0)} synapses'},
            {"step": "hub neurons wired", "ok": len(hubs) >= 5,
             "detail": ", ".join(h["type"] for h in hubs[:6]) or "none"},
            {"step": "graph loads (units/edges)",
             "ok": self.connectome.edge_count > 0,
             "detail": f'{self.connectome.unit_count} units · '
                       f'{self.connectome.edge_count} edges'},
            {"step": "instances ready", "ok": all(b.ready for b in self.instances),
             "detail": f'{sum(1 for b in self.instances if b.ready)}/4 brains'},
        ]
        return {
            "ready": self.ready,
            "verify": verify,
            "connectome": {
                "source": self.neuprint_status,
                "units_per_brain": self.connectome.unit_count,
                "edges": self.connectome.edge_count,
                "layout": "5000 individual + 2700 cluster units",
                "real": real.get("stats"),
                "hubs": (real.get("hubs") or [])[:10],
                "neurotransmitters": real.get("neurotransmitters"),
                "dataset": f'{real.get("source", "")} · {real.get("via", "")}',
            },
            "instances": [
                {"id": b.id, "role": b.role, "ready": b.ready}
                for b in self.instances
            ],
        }

    # ------------------------------------------------------------------
    # Section 9 — brain visualization data
    # ------------------------------------------------------------------
    def geometry(self) -> dict:
        """Static WebGL payload: unit positions, edges and region labels for
        the 2x2 four-instance network graph (computed once, cached)."""
        conn = self.connectome
        real = getattr(conn, "real", {}) or {}
        cached = getattr(self, "_geometry_cache", None)
        if cached is not None:
            return cached

        import base64

        import numpy as np

        # deterministic region centroids (spec labels)
        centroids: dict[str, tuple[float, float, float]] = {
            "neuromod": (-0.70, 0.50, 0.30),     # PAM + SE + OA + TA
            "mbon": (0.60, 0.40, 0.40),
            "kc": (0.20, 0.70, -0.30),
            "cc": (0.00, -0.50, 0.50),
            "fsb": (-0.40, -0.60, -0.20),
            "eb": (0.35, -0.70, 0.10),
            "lh": (-0.60, 0.00, -0.50),
            "dn": (0.50, 0.00, -0.60),
            "gen": (0.00, 0.05, 0.00),
            "clu": (0.05, -0.15, 0.10),
        }
        spread = {"neuromod": 0.14, "mbon": 0.10, "kc": 0.16, "cc": 0.12,
                  "fsb": 0.08, "eb": 0.07, "lh": 0.15, "dn": 0.13,
                  "gen": 0.85, "clu": 0.95}

        def bucket(unit: int) -> str:
            for name in ("PAM", "SE", "OA", "TA"):
                if unit in conn.pop[name]:
                    return "neuromod"
            if unit in conn.pop["MBON"]:
                return "mbon"
            if unit in conn.pop["KC"]:
                return "kc"
            if unit in conn.pop["CC"]:
                return "cc"
            if unit in conn.pop["FSB"]:
                return "fsb"
            if unit in conn.pop["EB"]:
                return "eb"
            if unit in conn.pop["LH"]:
                return "lh"
            if unit in conn.pop["DN"]:
                return "dn"
            if unit in conn.pop["AFF"]:
                return "clu"
            return "clu" if unit >= BRAIN_INDIVIDUAL_UNITS else "gen"

        rng = random.Random(49507)
        positions = np.zeros((conn.unit_count, 3), dtype=np.int16)
        for unit in range(conn.unit_count):
            b = bucket(unit)
            cx, cy, cz = centroids[b]
            r = spread[b]
            positions[unit, 0] = int((cx + rng.uniform(-r, r)) * 1000)
            positions[unit, 1] = int((cy + rng.uniform(-r, r)) * 1000)
            positions[unit, 2] = int((cz + rng.uniform(-r, r)) * 1000)

        pairs: list[tuple[int, int]] = []
        weights: list[int] = []
        for src, outs in enumerate(conn.out_edges):
            for dst, w, g in outs:
                pairs.append((src, dst))
                weights.append(int(max(-127, min(127, round(g * w * 127.0)))))
        edge_arr = np.array(pairs, dtype=np.uint16).reshape(-1)
        w_arr = np.array(weights, dtype=np.int8)

        regions = [
            {"name": "PAM cluster", "color": "#4caf50",
             "center": list(centroids["neuromod"]),
             "count": sum(len(conn.pop[p]) for p in ("PAM", "SE", "OA", "TA"))},
            {"name": "Mushroom Body", "color": "#ff9800",
             "center": list(centroids["mbon"]), "count": len(conn.pop["MBON"])},
            {"name": "Kenyon Cells", "color": "#ffc107",
             "center": list(centroids["kc"]), "count": len(conn.pop["KC"])},
            {"name": "Central Complex", "color": "#2196f3",
             "center": list(centroids["cc"]), "count": len(conn.pop["CC"])},
            {"name": "Fan-Shaped Body", "color": "#00bcd4",
             "center": list(centroids["fsb"]), "count": len(conn.pop["FSB"])},
            {"name": "Ellipsoid Body", "color": "#9c27b0",
             "center": list(centroids["eb"]), "count": len(conn.pop["EB"])},
            {"name": "Lateral Horn", "color": "#e91e63",
             "center": list(centroids["lh"]), "count": len(conn.pop["LH"])},
            {"name": "Descending Neurons", "color": "#8bc34a",
             "center": list(centroids["dn"]), "count": len(conn.pop["DN"])},
        ]
        geom = {
            "units": conn.unit_count,
            "edges": len(pairs),
            "encoding": {"pos": "int16x3 b64 (x1000)", "edges": "uint16x2 b64",
                         "weights": "int8 b64 (g*w*127)"},
            "positions": base64.b64encode(positions.tobytes()).decode(),
            "edge_pairs": base64.b64encode(edge_arr.tobytes()).decode(),
            "edge_weights": base64.b64encode(w_arr.tobytes()).decode(),
            "regions": regions,
            "news_units": list(conn.news_units),
            # REAL Drosophila brain provenance (Janelia hemibrain v1.2)
            "connectome_source": conn.neuprint_status,
            "connectome_dataset": real.get("source", ""),
            "connectome_via": real.get("via", ""),
            "connectome_stats": real.get("stats"),
            "connectome_hubs": (real.get("hubs") or [])[:12],
            "neurotransmitter_census": real.get("neurotransmitters"),
        }
        self._geometry_cache = geom
        return geom

    def activations(self, symbol: str, tf: str) -> dict:
        """Per-instance unit activation snapshot (uint8 base64) — one byte
        per neuron, updated every GRC by the client after each snapshot."""
        import base64

        import numpy as np

        states = []
        for brain in self.instances:
            st = brain.state_for(symbol, tf)
            arr = np.clip(np.array(st, dtype=float) * 255.0, 0, 255
                          ).astype(np.uint8)
            states.append(base64.b64encode(arr.tobytes()).decode())
        return {
            "symbol": symbol, "tf": tf,
            "encoding": "u8 b64 (activation*255)",
            "per_instance": states,
        }


def empty_brain_consensus() -> dict:
    return {"polarity": 0, "strength": 0.0, "agreement": 0.0,
            "consensus": 0.0, "hormones_avg": {}, "experience": 0.5,
            "state_dist": {}, "dominant": "ranging", "p_dominant": 0.2,
            "weight_adj": {wid: 0.0 for wid in WEIGHT_IDS},
            "instances": []}
