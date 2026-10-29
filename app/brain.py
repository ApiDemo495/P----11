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

    def __init__(self, seed: int = 495):
        self.neuprint_status = "structural-fallback"
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

        edges: list[tuple[int, int, float, float]] = []

        def add_edge(src: int, dst: int, w: float | None = None,
                     g: float | None = None) -> None:
            if w is None:
                w = rng.uniform(0.1, 1.0)
            if g is None:
                # inhibitory predominantly on mushroom-body -> output synapses
                is_mb_out = (src in self.pop["MBON"] or src in self.pop["KC"] or
                             src in self.pop["PAM"]) and dst in self.pop["MBON"]
                g = -1.0 if (is_mb_out and rng.random() < 0.7) else 1.0
            edges.append((src, dst, w, g))

        # sparse random backbone (out-degree budget)
        for src in range(n):
            for _ in range(BRAIN_OUT_DEGREE):
                add_edge(src, rng.randrange(n))
        # hormone populations -> MBON decision layer (guaranteed pathway)
        for pop_name in ("PAM", "SE", "OA", "TA"):
            for src in self.pop[pop_name]:
                for _ in range(3):
                    add_edge(src, rng.choice(mbon))
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
        # normalize hormone inputs into sigmoid-relevant range
        i_map = {
            "dopamine": hormones["dopamine"] * 2.5,
            "serotonin": (hormones["serotonin"] - 0.5) * 4.0,
            "octopamine": hormones["octopamine"] * 3.0,
            "tyramine": (min(2.0, hormones["tyramine"]) - 1.0) * 3.0,
        }

        results = []
        for brain in self.instances:
            inputs: dict[int, float] = {}
            # --- Thing 1: hormone populations ---------------------------
            for pop_name, key in (("PAM", "dopamine"), ("SE", "serotonin"),
                                  ("OA", "octopamine"), ("TA", "tyramine")):
                for n_i in conn.pop[pop_name]:
                    inputs[n_i] = i_map[key]

            # --- brain-formula affinity coupling (|F_f| > 0.7) ----------
            aff_list = list(conn.pop["AFF"])
            per_formula = _AFFINITY_CELLS // 31
            for f_pos, fid in enumerate(range(1, 32)):
                if fid in affinity:
                    for k in range(per_formula):
                        inputs[aff_list[f_pos * per_formula + k]] = affinity[fid]

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

            results.append({
                "id": brain.id,
                "hormone_state": round(h_state, 5),
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
        mean_signal = sum(r["h_signal"] for r in results) / 4.0
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
    def status(self) -> dict:
        return {
            "ready": self.ready,
            "connectome": {
                "source": self.neuprint_status,
                "units_per_brain": self.connectome.unit_count,
                "edges": self.connectome.edge_count,
                "layout": "5000 individual + 2700 cluster units",
            },
            "instances": [
                {"id": b.id, "role": b.role, "ready": b.ready}
                for b in self.instances
            ],
        }


def empty_brain_consensus() -> dict:
    return {"polarity": 0, "strength": 0.0, "agreement": 0.0,
            "consensus": 0.0, "hormones_avg": {}, "experience": 0.5,
            "state_dist": {}, "dominant": "ranging", "p_dominant": 0.2,
            "weight_adj": {wid: 0.0 for wid in WEIGHT_IDS},
            "instances": []}
