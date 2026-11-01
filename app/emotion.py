"""Section 5 — EMOTIONAL FLOW (E1..E22).

Twenty-two dimensions with exact coefficient sets from the specification:

  E1  FII   Fear Intensity Index          E12 DRD   Daily Range Deviation
  E2  GSL   Guardian-of-Loss Syndrome     E13 ERI   Extreme Reversal Index
  E3  FPR   Fear Put Ratio                E14 OFS   Optimism/Fear Sentiment
  E4  PCP   Panic Coefficient (rising)    E15 IAF   Intent/Action Fraction
  E5  CD    Octopamine Reflex (original)  E16 LAI   Lead/Announce Impulse
  E6  ESI   Endorphin Session Intensity   E17 CDP   Crowd-Directional Pull
  E7  DPI   Dopamine (smart money)        E18 HTI   Honeymoon Trend Inversion
  E8  ATR   Volume-ATR Emotion            E19 SMDM  Smart-Money Divergence
  E9  RTP   Roll-over Trauma/Panic        E20 NMS   News/Momentum Surge
  E10 ABS   Absorption by smart money     E21 TEDR  Trend Exhaustion Decay
  E11 HMC   Herd/Mirror Co-movement       E22 CERC  Composite (weighted sum)

Every E1..E21 output is a sigmoid/fraction in [0,1]; each contributes to CERC
as  w_e * E_e * direction_e  with the TEDR participation multiplier
(1 - 0.5 * E21) on the TEDR term.  E22 CERC weights start at 1/21 and learn
with the same eta=0.02 clamp/renormalize reward rule as the CCI weights, and
CERC feeds the CCI as weighted component id 98.
"""
from __future__ import annotations

import math

from .config import HIST_EMOTION_LAG
from .models import FormulaContext
from .formulas._helpers import pearson, sign, std_pop

ETA = 0.015               # Section 10: η_emotion — floor (see COLD rate below)
W_CLAMP_LO = 0.005        # widened: predictive channels can dominate
W_CLAMP_HI = 0.30
ETA_COLD = 0.08           # cold-start rate — a fresh brain learns FAST
COLD_HALFLIFE = 150
ACC_SMOOTH = 0.90
EMO_IDS = list(range(1, 22))          # CERC is E22 = the composite itself
CERC_WEIGHT_ID = 22


def sig(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lag_value(ring) -> float | None:
    """GRC-lagging comparator: the value from HIST_EMOTION_LAG GRCs ago
    (oldest ring entry once full; the most recent entry during bootstrap)."""
    if len(ring) == 0:
        return None
    if len(ring) >= HIST_EMOTION_LAG:
        return ring.values()[0]
    return ring.last()


def _neg_frac(closes: list[float], n: int) -> float:
    seg = closes[-n:]
    if len(seg) < 2:
        return 0.0
    rets = [(seg[i] - seg[i - 1]) / seg[i - 1] for i in range(1, len(seg))]
    if not rets:
        return 0.0
    return sum(1 for r in rets if r < 0) / len(rets)


def _atr5(ohlc: dict | None, closes: list[float]) -> float:
    """ATR_5: average true range over the last 5 minutes."""
    if ohlc and len(ohlc.get("highs", [])) >= 6:
        h, l, c = ohlc["highs"], ohlc["lows"], ohlc["closes"]
        trs = []
        for i in range(-5, 0):
            pc = c[i - 1]
            trs.append(max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc)))
        return max(1e-9, sum(trs) / 5)
    if len(closes) < 6:
        return 1e-9
    trs = [abs(closes[i] - closes[i - 1]) for i in range(-5, 0)]
    return max(1e-9, sum(trs) / 5)


class EmotionSystem:
    """Holds the 21 CERC weights per (symbol, timeframe) and computes E1..E22."""

    def __init__(self) -> None:
        self.weights: dict[tuple[str, str], list[float]] = {}
        self.acc: dict = {}        # (sym, tf) -> 21 EWMA accuracies
        self.n_updates: dict = {}  # (sym, tf) -> reward-call count

    @property
    def ready(self) -> bool:
        return True

    def initialize(self) -> None:
        """No external dependencies (history rings bootstrap themselves)."""
        return

    def reset_weights(self) -> None:
        self.weights.clear()

    # ------------------------------------------------------------------
    def _weights(self, symbol: str, tf: str) -> list[float]:
        key = (symbol, tf)
        w = self.weights.get(key)
        if w is None:
            w = [1.0 / 21.0] * 21
            self.weights[key] = w
        return w

    def _accuracy(self, symbol: str, tf: str) -> list[float]:
        """EWMA predictive accuracy per CERC channel (0.5 = uninformative)."""
        key = (symbol, tf)
        a = self.acc.get(key)
        if a is None:
            a = [0.5] * 21
            self.acc[key] = a
        return a

    # ------------------------------------------------------------------
    def compute(
        self,
        ctx: FormulaContext,
        market: dict,
        brain_res: dict,
        news_agg: dict,
        prev_cerc: float | None,
    ) -> dict:
        cl = ctx.P                                  # timeframe window closes
        mc = market["closes"]                       # long window (>= 60+ mins)
        open_ = market.get("session_open") or mc[0]
        live = market.get("live", cl[-1])

        # TPDI fallback: direction = window displacement sign
        trend_w = sign(cl[-1] - cl[0]) if cl[0] else 0
        trend10 = sign(mc[-1] - mc[-11]) if len(mc) > 11 else trend_w
        r30 = (mc[-1] / mc[-31] - 1.0) if len(mc) > 31 else (
            (cl[-1] / cl[0] - 1.0) if cl[0] else 0.0
        )
        rets = [(mc[i] - mc[i - 1]) / mc[i - 1] for i in range(1, len(mc))]
        s1 = std_pop(rets[-10:]) if len(rets) >= 2 else 0.0
        s30 = std_pop(rets[-min(30, len(rets)):]) if len(rets) >= 2 else 0.0
        vols = market.get("volumes", [0.0] * len(mc))
        v10 = sum(vols[-10:]) / min(10, len(vols)) if vols else 0.0
        v30 = sum(vols[-30:]) / min(30, len(vols)) if vols else 0.0
        v5 = sum(vols[-5:]) / min(5, len(vols)) if vols else 0.0
        vall = sum(vols) / len(vols) if vols else 0.0
        hist = ctx.hist
        emos: dict[int, float] = {}
        dirs: dict[int, int] = {}

        # ---- E1 FII: fear intensity index --------------------------------
        streak = 0
        for p in reversed(mc):
            if p > open_:
                streak += 1
            else:
                break
        dpi_ring = hist.ring(hist.tf_key(ctx.symbol, ctx.tf, "dpi"), HIST_EMOTION_LAG)
        dpi_mean = dpi_ring.mean() if len(dpi_ring) else 0.0
        tau_half = ctx.intermediates.get("tau")
        if tau_half is None or not (tau_half > 0):
            tau_half = 60.0
        tau_eff = max(1.0, tau_half * (1.0 - 0.5 * dpi_mean))
        pdi_fii = 1.0 - math.exp(-3.0 * (streak / tau_eff))
        emos[1] = sig(5.0 * (pdi_fii - 0.5))
        if pdi_fii >= 0.85:
            dirs[1] = -1
        elif pdi_fii <= 0.15:
            dirs[1] = 1
        else:
            dirs[1] = trend_w

        # ---- E2 GSL: guardian-of-loss syndrome ---------------------------
        p_hit = hist.p_hit(ctx.symbol)
        p_min60 = min(mc[-min(60, len(mc)):])
        atr5 = _atr5(market.get("ohlc"), mc)
        pen = p_hit + (open_ - p_min60) / atr5
        emos[2] = sig(-4.0 * (pen - 1.0))
        dirs[2] = 1 if emos[2] > 0.5 else -1

        # ---- E3 FPR: fear put ratio -------------------------------------
        ratio = (s1 / s30) if s30 > 1e-12 else 1.0
        emos[3] = sig(-5.0 * (1.0 - ratio - r30))
        dirs[3] = -sign(r30 + (ratio - 1.0)) if (r30 or ratio != 1.0) else 0

        # ---- E4 PCP: panic coefficient (rising) --------------------------
        v10_30 = (v10 / v30) if v30 > 0 else 1.0
        emos[4] = sig(6.0 * (ratio - 1.0))          # sigma expanding
        _ = v10_30
        dirs[4] = -sign(trend_w)

        # ---- E5 CD: octopamine reflex (original variant) -----------------
        octo_raw = market.get("octo_move")
        if octo_raw is None:
            octo_raw = rets[-1] if rets else 0.0
        volchg = vols[-1] / v10 if v10 > 0 else 1.0
        octo_now = sig(4.0 * octo_raw * (1.0 + max(0.0, volchg - 1.0)))
        octo_ring = hist.ring(hist.tf_key(ctx.symbol, ctx.tf, "octo"), HIST_EMOTION_LAG)
        octo_lag_v = _lag_value(octo_ring)
        octo_lag = sig(4.0 * octo_lag_v) if octo_lag_v is not None else 0.5
        emos[5] = 1.0 if (octo_now >= 2.0 * octo_lag and octo_now > 0.5) else 0.0
        dirs[5] = 1 if emos[5] > 0.5 else -1        # fear reflex -> buy bias

        # ---- E6 ESI: endorphin session intensity -------------------------
        p_min_floor = max(p_min60, open_ * 0.995)
        p_max_sess = max(market.get("session_high", p_min_floor), live, p_min_floor)
        pdi_sess = ((live - p_min_floor) / (p_max_sess - p_min_floor)
                    if p_max_sess > p_min_floor else 0.5)
        # brain->emotion hormone averaging: averaged serotonergic tone
        # (complacency) modulates the session endorphin intensity
        serotonin_brain = brain_res.get("hormones_avg", {}).get("serotonin", 0.5)
        emos[6] = sig(10.0 * (pdi_sess - 0.5) + 3.0 * (serotonin_brain - 0.5))
        dirs[6] = -sign(live - open_) if live != open_ else 0

        # ---- E7 DPI: dopamine / smart money buying -----------------------
        dopamine_brain = brain_res.get("hormones_avg", {}).get("dopamine", 0.0)
        dpi_raw = (v10_30 - 1.0) + r30 - 0.1        # excess volume + move - 0.1
        # brain->emotion hormone averaging: averaged dopaminergic input
        dpi_raw += 0.5 * (min(1.0, dopamine_brain) - 0.5)
        emos[7] = sig(6.0 * dpi_raw)
        dirs[7] = 1 if emos[7] > 0.5 else -1

        # ---- E8 ATR_emotion: volume climax -------------------------------
        prior = vols[:-1]
        if prior:
            max_prior = max(prior[-min(240, len(prior)):])
            if max_prior > 0 and vols[-1] >= 2.0 * max_prior:
                emos[8] = 1.0
            else:
                emos[8] = sig(4.0 * (vols[-1] / max_prior - 1.5)) if max_prior > 0 else 0.5
        else:
            emos[8] = 0.5
        dirs[8] = 0                                 # magnitude channel

        # ---- E9 RTP: roll-over trauma / panic ----------------------------
        negf = _neg_frac(mc, 10)
        hit5 = hist.sl_hit_ratio_recent(ctx.symbol)
        comps = [negf]
        if len(hist.sl_hit_events.get(ctx.symbol, [])) >= 3:
            comps.append(hit5)
        emos[9] = sum(comps) / len(comps)
        dirs[9] = 1 if emos[9] > 0.5 else -1         # pain -> contrarian buy

        # ---- E10 ABS: absorption by smart money --------------------------
        btc = market.get("btc_closes") or mc
        r_btc20 = (btc[-1] / btc[-min(21, len(btc))] - 1.0) if len(btc) > 5 else 0.0
        emos[10] = sig(2.0 * r_btc20)
        if abs(r_btc20) > 0.01:
            dirs[10] = 1
        elif abs(r_btc20) < 0.001:
            dirs[10] = 0
        else:
            dirs[10] = -sign(r_btc20)

        # ---- E11 HMC: herd / mirror co-movement --------------------------
        peer = market.get("peer_returns") or []
        agreements = []
        for row in peer[-10:]:
            if len(row) >= 2:
                same = 0
                total = 0
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        if row[a] == 0 or row[b] == 0:
                            continue
                        total += 1
                        if (row[a] > 0) == (row[b] > 0):
                            same += 1
                if total:
                    agreements.append(same / total)
        agree_ratio = sum(agreements) / len(agreements) if agreements else 0.5
        emos[11] = sig(6.0 * (agree_ratio - 0.5))
        if emos[11] > 0.9:
            dirs[11] = -trend10
        else:
            dirs[11] = trend10

        # ---- E12 DRD: daily range deviation ------------------------------
        emos[12] = min(1.0, abs(r30))
        dirs[12] = -1 if r30 > 0 else (1 if r30 < 0 else 0)   # spec: overbought/oversold

        # ---- E13 ERI: extreme reversal index -----------------------------
        rts = [(cl[i] - cl[i - 1]) / cl[i - 1] for i in range(1, len(cl))] if len(cl) > 2 else []
        flips = sum(1 for i in range(1, len(rts))
                    if rts[i] * rts[i - 1] < 0 and rts[i] != 0)
        run_len = 0
        for r in reversed(rts):
            if r == 0 or (rts[-1] and r * rts[-1] > 0):
                run_len += 1
            else:
                break
        # long-run interruptions: sign flips after runs of >= 3
        interruptions = 0
        cur_run = 1
        for i in range(1, len(rts)):
            if rts[i] * rts[i - 1] > 0:
                cur_run += 1
            else:
                if cur_run >= 3:
                    interruptions += 1
                cur_run = 1
        abs_rts = [abs(r) for r in rts]
        last_abs = abs_rts[-1] if abs_rts else 0.0
        ordered = sorted(abs_rts, reverse=True)
        rank = (ordered.index(last_abs) + 1) if last_abs in ordered else len(ordered)
        top5_val = max(0.0, 1.0 - (rank - 1) / 5.0) if ordered else 0.0
        extreme_count = sum(1 for r in abs_rts if s1 > 1e-12 and r > 3 * s1)
        r_win = cl[-1] / cl[0] - 1.0 if cl[0] else 0.0
        flags = [
            (0.4537, (flips - 2) / 2.0),                 # major reversal >= 2
            (0.1007, (interruptions - 1) / 1.0),         # interruption >= 1
            (0.1734, (top5_val - 0.6) / 0.4),            # within top-5 move
            (0.1734, (run_len - 10) / 5.0),              # stall >= 10 min
            (0.0641, (extreme_count - 3) / 1.5),         # extremes >= 3
            (0.0346, (abs(r_win) - 0.3) / 0.15),         # distance >= 30%
        ]
        wsum = sum(w for w, _ in flags)
        emos[13] = sum(w * sig(2.0 * v) for w, v in flags) / wsum
        dirs[13] = -sign(trend_w) if emos[13] >= 0.3 else 0

        # ---- E14 OFS: optimism / fear sentiment --------------------------
        if len(cl) > 1:
            emos[14] = sum(1 for i in range(1, len(cl)) if cl[i] > cl[i - 1]) / (len(cl) - 1)
        else:
            emos[14] = 0.5
        dirs[14] = 1 if emos[14] > 0.5 else -1

        # ---- E15 IAF: intent / action fraction ---------------------------
        intent = (v5 / vall) if vall > 0 else 1.0
        emos[15] = _clamp(intent - 0.5, 0.0, 1.0)
        dirs[15] = sign(cl[-1] - cl[0]) if cl[0] else 0   # TPDI fallback

        # ---- E16 LAI: lead / announce impulse ----------------------------
        vel = (rets[-1] / s1) if (rets and s1 > 1e-12) else 0.0
        emos[16] = sig(3.0 * vel + 10.0 * (pdi_sess - 0.65))
        dirs[16] = 1 if emos[5] >= 0.7 else -1         # spec: CD >= 0.7 -> +1

        # ---- E17 CDP: crowd-directional pull -----------------------------
        pol = news_agg.get("polarity", 0)
        count = news_agg.get("count", 0)
        emos[17] = sig(6.0 * pol) if count else 0.5
        if not count:
            dirs[17] = 0
        else:
            dirs[17] = 1 if pol > 0 else (-1 if pol < 0 else 0)

        # ---- E18 HTI: honeymoon trend inversion --------------------------
        pidx_ring = hist.ring(hist.tf_key(ctx.symbol, ctx.tf, "pidx"), HIST_EMOTION_LAG)
        lo30 = min(mc[-min(30, len(mc)):])
        hi30 = max(mc[-min(30, len(mc)):])
        idx_now = ((live - lo30) / (hi30 - lo30)) if hi30 > lo30 else 0.5
        idx_lag = _lag_value(pidx_ring)
        delta = (idx_now - idx_lag) if idx_lag is not None else 0.0
        emos[18] = min(1.0, abs(delta) * 4.0)
        dirs[18] = -sign(delta) if abs(delta) > 0.02 else 0

        # ---- E19 SMDM: smart-money divergence ----------------------------
        cmp_div = market.get("cmp_divergence", 0.0)   # Part-3 seam (orderflow)
        hmc = emos[11]
        emos[19] = sig(6.0 * ((1.0 - hmc) * cmp_div - 0.5))
        dirs[19] = 1 if emos[19] > 0.5 else -1         # spec: sign(E-0.5)

        # ---- E20 NMS: news / momentum surge ------------------------------
        pol_now = emos[17] * (1 if pol >= 0 else -1)
        nsp_ring = hist.ring(hist.tf_key(ctx.symbol, ctx.tf, "nsp"), HIST_EMOTION_LAG)
        pol_lag_v = _lag_value(nsp_ring)
        pol_lag = pol_lag_v if pol_lag_v is not None else 0.5
        accel = pol_now - pol_lag
        corr = market.get("news_price_corr", 0.0)      # Part-3 seam
        emos[20] = sig(6.0 * (accel + max(0.0, corr)))
        dirs[20] = trend10 if emos[20] > 0.6 else 0    # spec: >0.6 else 0

        # ---- E21 TEDR: trend exhaustion decay ratio ----------------------
        d1 = mc[-1] - mc[-6] if len(mc) > 6 else 0.0
        d2 = mc[-6] - mc[-11] if len(mc) > 11 else 0.0
        denom = abs(d2) + abs(d1)
        decay = (abs(d2) - abs(d1)) / denom if denom > 1e-12 else 0.0
        emos[21] = sig(6.0 * decay)                    # decelerating momentum
        dirs[21] = -sign(trend10) if emos[21] > 0.6 else 0

        # ---- E22 CERC: weighted composite --------------------------------
        w = self._weights(ctx.symbol, ctx.tf)
        cerc = 0.0
        for i, eid in enumerate(EMO_IDS):
            term = w[i] * emos[eid] * dirs[eid]
            if eid == 21:
                term *= (1.0 - 0.5 * emos[21])         # TEDR participation mult
            cerc += term

        # GRC-lag rings (written by engine when record=True)
        lag_rings = {
            "dpi": dpi_raw,
            "octo": octo_raw * (1.0 + max(0.0, volchg - 1.0)),
            "pidx": idx_now,
            "nsp": emos[17] * (1 if pol >= 0 else -1),
        }
        return {
            "emotions": {k: round(v, 5) for k, v in emos.items()},
            "directions": dirs,
            "cerc": round(cerc, 5),
            "rtp": emos[9],
            "tedr": emos[21],
            "prev_cerc": prev_cerc,
            "lag_values": lag_rings,
        }

    # ------------------------------------------------------------------
    def reward_update(self, symbol: str, tf: str, emos: dict[int, float],
                      dirs: dict[int, int], cerc_before: float,
                      move: float, eta: float | None = None,
                      scale: float | None = None) -> None:
        """GRC-cadence auto-learning for the 21 CERC weights (eta rule).

        Accuracy-driven upgrades (user directive: weightage follows
        predictive accuracy):
          * COLD-START eta — rate decays from ETA_COLD to the floor with a
            150-update half-life, so the first few hundred cycles actually
            differentiate channels instead of freezing at uniform.
          * MAGNITUDE reward — a big verified move teaches more than noise:
            reward = ±clip(|move|/scale, 0.3, 1.5).
          * per-channel EWMA ACCURACY exposed to the UI/API.
        """
        base = ETA if eta is None else eta
        key = (symbol, tf)
        n = self.n_updates.get(key, 0)
        rate = max(base, ETA_COLD * 0.5 ** (n / COLD_HALFLIFE))
        self.n_updates[key] = n + 1
        w = self._weights(symbol, tf)
        acc = self._accuracy(symbol, tf)
        move_dir = sign(move)
        mag = 1.0
        if scale and move != 0.0:
            mag = _clamp(abs(move) / scale, 0.3, 1.5)
        for i, eid in enumerate(EMO_IDS):
            term = w[i] * emos.get(eid, 0.0) * dirs.get(eid, 0)
            if eid == 21:
                term *= (1.0 - 0.5 * emos.get(21, 0.0))
            if move_dir == 0:
                reward = 0.0
            else:
                correct = sign(term) == move_dir
                reward = mag if correct else -mag
                acc[i] = ACC_SMOOTH * acc[i] + (1 - ACC_SMOOTH) * (1.0 if correct else 0.0)
            # same multiplicative eta rule as the CCI weight learning:
            # omega(t) = omega(t-1) * (1 + eta * reward), then clamp
            w[i] = _clamp(w[i] * (1.0 + rate * reward), W_CLAMP_LO, W_CLAMP_HI)
        total = sum(w)
        if total > 0:
            for i in range(21):
                w[i] /= total
        _ = cerc_before

    # ------------------------------------------------------------------
    def status(self) -> dict:
        out: dict[str, list] = {}
        for (sym, tf), w in self.weights.items():
            out[f"{sym}:{tf}"] = [round(x, 4) for x in w]
        return out

    def accuracy(self) -> dict:
        out: dict[str, list] = {}
        for (sym, tf), a in self.acc.items():
            out[f"{sym}:{tf}"] = [round(x, 4) for x in a]
        return out
