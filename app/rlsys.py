"""Reinforcement-learning controllers (user: 'use RL to find & fix errors,
boost prediction accuracy, fix every formula/emotion/physics').

Six controllers, one console — all learning ONLY from real settled
windows (strict-sign reward: win +1, loss −1, no-trade/skip 0):

  1. ERROR-FINDER (auto-fix + rollback): every graded component already
     has reward rings (facc/eacc/pacc strict-sign outcomes). When a
     component's pooled reward drops below0.40 over n≥8 windows the
     controller DAMPS its effective weight (×0.75, floor0.4), watches the
     next windows, RESTORES the weight when reward recovers (≥0.50),
     and deepens (or rolls back) otherwise. Every action is recorded —
     visible in /api/rl/status — so RL fixes are auditable, never
     invisible magic.
  1b. WINNER BOOST (the symmetric half): reward >0.60 over n≥8 earns
     weight ×1.10 per sustained pass up to a ×1.5 cap; the boost is
     withdrawn the moment reward falls back under0.50, and a component
     that turns bad loses its boost BEFORE any damp is applied. Winners
     earn weight — they are never just 'not losers'.
  2. ISSUE-THRESHOLD BANDIT (accuracy booster): a5-armed bandit over the
     confidence issue threshold {0.25,0.30,0.35,0.40,0.45}
     commits to a bucket for8 settled windows, scores it by realised
     strict-sign reward (issued: ±1, no-trade:0), re-picks
     argmax(mean)+15% exploration. The threshold only moves where the
     DATA says issued trades win more.
  2b. CONTEXTUAL BANDIT: the same bandit kept PER CONTEXT
     (asset|timeframe|volatility-regime) once a context has its own
     settled history; unseen contexts fall back to the global bucket, so
     a hot BTC/tf1 regime no longer drags the threshold for a quiet
     BNB/tf4 — and no context invents a threshold before it has data.
  3. Q-TABLE over window states: state = volatility regime × trend/chop
     (measured from the tape at compute time), action = one of five
     subsystem-mix presets applied to the CCI shares. At settlement the
     (state, action) pair that was frozen into the lock is updated
     Q += α(r − Q), α=0.25, on decided windows only. ε-greedy (15%)
     selection; a state with no decided data yet plays 'balanced' —
     exactly the legacy mix — never a guessed mix.
  4. TARGET TRACKER (honest): measures pooled ledger accuracy against
     the user's90% target and prints the REAL number and gap — the RL
     loop climbs toward the target; it never rewrites history to fake
     reaching it.

Persistence: data/rl.json with BUILD_VERSION guard (stale state dropped).
Fairness: a fresh process starts at threshold0.35, factor1.0 and the
balanced mix — exactly the legacy behaviour, so pinned confidence tests
stay green until real settlements move the learners.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from typing import Any

BUILD_VERSION = 2
RL_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "rl.json")
BUCKETS = [0.25, 0.30, 0.35, 0.40, 0.45]
COMMIT_N = 8             # settled windows per bandit commitment
EXPLORE = 0.15           # chance to explore another bucket at re-pick
DAMP_FLOOR = 0.4
DAMP_STEP = 0.75         # multiply down by25% per sustained offence
REWARD_LOW = 0.40
REWARD_OK = 0.50
REWARD_HIGH = 0.60       # winner-boost entry bar
BOOST_STEP = 1.10        # multiply up by10% per sustained win streak
BOOST_CAP = 1.5
MIN_N = 8
ACTION_CAP = 200
ACC_TARGET = 0.90        # user target — reported honestly, never faked

# ---- Q-table over window states -------------------------------------
# Actions are MULTIPLIERS over the subsystem shares (renormalised to
# 1.0 when applied). 'balanced' is the identity — the legacy mix.
MIX_ACTIONS: dict[str, dict[str, float]] = {
    "balanced": {},
    "core-heavy": {"core": 1.30, "brain": 0.8, "emotion": 0.8,
                   "physics": 0.9, "news": 0.8, "hf": 0.9, "tracer": 0.9},
    "physics-heavy": {"physics": 1.35, "core": 0.9, "emotion": 0.85,
                      "news": 0.85},
    "sentiment-heavy": {"emotion": 1.30, "news": 1.25, "core": 0.9,
                        "physics": 0.9},
    "brain-heavy": {"brain": 1.40, "core": 0.9, "emotion": 0.9},
}
Q_ALPHA = 0.25           # one-step Q learning rate (γ=0: window reward
                         # is fully realised at settlement, no bootstrap)
Q_MIN_N = 3              # decided samples before a state leaves balanced


def window_state(sigma_ratio: float, trend_strength: float) -> str:
    """Window state for the Q-table, from measured tape quantities:
    sigma_ratio = recent vol / own baseline (>1.25 high, <0.8 low);
    trend_strength = |drift| in recent-sigma units (≥1.0 = trend)."""
    vol = ("highvol" if sigma_ratio > 1.25 else
           "lowvol" if sigma_ratio < 0.80 else "midvol")
    return f"{vol}-{'trend' if trend_strength >= 1.0 else 'chop'}"


def context_key(symbol: str, tf: str, regime: str) -> str:
    """Contextual-bandit context: asset|timeframe|volatility regime."""
    return f"{symbol}|{tf}|{regime}"


def _pool_mean(engine: Any, ring_name: str) -> tuple[float, int]:
    """Mean of a strict-sign ring pooled over every (asset, tf)."""
    from .config import SYMBOL_ORDER, TF_ORDER  # local import: no cycle
    vals: list[float] = []
    for sym in SYMBOL_ORDER:
        for tf in TF_ORDER:
            try:
                vals.extend(float(x) for x in engine.hist.ring(
                    engine.hist.tf_key(sym, tf, ring_name), 50))
            except Exception:
                continue
    if not vals:
        return 0.0, 0
    return sum(vals) / len(vals), len(vals)


def _mean_across(engine: Any, names: list[str]) -> tuple[float, int]:
    means = []
    ns = 0
    for nm in names:
        m, n = _pool_mean(engine, nm)
        if n >= MIN_N:
            means.append(m)
            ns += n
    if not means:
        return 0.0, 0
    return sum(means) / len(means), ns


def damp_all(weights: dict[int, float]) -> dict[int, float]:
    """Apply active error-finder damp factors to a weight dict."""
    rl = get_rl()
    if not rl.damp:
        return weights
    return {fid: v * rl.damp.get(int(fid), 1.0)
            for fid, v in weights.items()}


_SHARE_KEY_ID = {"brain": 97, "emotion": 98, "physics": 99,
                 "news": 100, "hf": 101, "tracer": 102}


def damp_shares(shares: dict[str, float]) -> dict[str, float]:
    """Subsystem-share layer: damp is applied to the share and the pile
    is renormalised to1.0 (same rule the learning layer uses)."""
    rl = get_rl()
    if not rl.damp and not rl.boost:
        return shares
    out: dict[str, float] = {}
    for k, v in shares.items():
        cid = _SHARE_KEY_ID.get(k)
        out[k] = float(v) * rl.factor(cid) if cid else float(v)
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


class RLSystem:
    def __init__(self, path: str | None = None) -> None:
        if path is None and os.environ.get("PYTEST_CURRENT_TEST"):
            import tempfile as _tf
            # per-PID file, removed on first use: a shared fixed tmp
            # path leaked learned damp/boost factors from one pytest
            # run into the next and silently re-weighted the CCI in
            # unrelated tests (the suite passed once, then failed).
            path = os.path.join(
                _tf.gettempdir(), f"predict4_test_rl_{os.getpid()}.json")
            try:
                os.remove(path)
            except OSError:
                pass
        self.path = path or RL_FILE
        self.damp: dict[int, float] = {}
        self.boost: dict[int, float] = {}
        self.active: dict[int, dict] = {}      # id -> damp action record
        self.boosted: dict[int, dict] = {}     # id -> boost action record
        self.actions: list[dict] = []
        self.bandit: dict[str, Any] = self._fresh_bandit()
        # contextual bandits: context -> bandit dict (own bucket/stats)
        self.contexts: dict[str, dict[str, Any]] = {}
        # Q-table: state -> action -> {"q": float, "n": int}
        self.q: dict[str, dict[str, dict[str, float]]] = {}
        self._q_pending: dict[str, dict[str, str]] = {}
        self.version = BUILD_VERSION
        self.load()

    @staticmethod
    def _fresh_bandit() -> dict[str, Any]:
        return {"bucket": 2,                       # default0.35
                "committed": 0,
                "stats": {str(i): {"n": 0, "reward": 0.0}
                          for i in range(len(BUCKETS))}}

    def factor(self, cid: int) -> float:
        """Learned weight factor for one component: damp × boost."""
        return float(self.damp.get(int(cid), 1.0)
                     * self.boost.get(int(cid), 1.0))

    # ------------------------------------------------------ persistence
    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            if int(d.get("build_version", -1)) != BUILD_VERSION:
                return                      # stale state dropped (B-8)
            self.damp = {int(k): float(v)
                         for k, v in (d.get("damp") or {}).items()}
            self.boost = {int(k): float(v)
                          for k, v in (d.get("boost") or {}).items()}
            self.active = {int(k): dict(v)
                           for k, v in (d.get("active") or {}).items()}
            self.boosted = {int(k): dict(v)
                            for k, v in (d.get("boosted") or {}).items()}
            self.actions = list(d.get("actions") or [])[-ACTION_CAP:]
            b = d.get("bandit") or {}
            if b.get("bucket") is not None:
                self.bandit = {
                    "bucket": int(b["bucket"]),
                    "committed": int(b.get("committed") or 0),
                    "stats": b.get("stats") or self.bandit["stats"],
                }
            self.contexts = {str(k): dict(v) for k, v in
                             (d.get("contexts") or {}).items()}
            self.q = {str(s): {str(a): {"q": float(v.get("q", 0.0)),
                                        "n": int(v.get("n", 0))}
                                for a, v in (acts or {}).items()}
                      for s, acts in (d.get("q") or {}).items()}
        except (OSError, ValueError, TypeError):
            pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(self.path) or ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"build_version": BUILD_VERSION,
                           "damp": self.damp, "boost": self.boost,
                           "active": self.active, "boosted": self.boosted,
                           "actions": self.actions[-ACTION_CAP:],
                           "bandit": self.bandit,
                           "contexts": self.contexts, "q": self.q}, fh)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # -------------------------------------------------- 1) error finder
    def audit(self, engine: Any) -> list[dict]:
        """Scan graded rings; damp broken components, restore healed ones.
        Returns the actions taken this pass (usually none)."""
        taken: list[dict] = []
        from .formulas import FORMULA_FUNCTIONS
        formula_ids = list(FORMULA_FUNCTIONS) + [20, 28, 30]
        targets: list[tuple[int, str, str]] = [
            (fid, "facc", "formula") for fid in formula_ids
        ] + [
            (98, "eacc", "emotion"),
            (99, "pacc", "physics"),
            (97, "facc_brain", "brain"),
            (102, "facc_tracer", "tracer"),
        ]
        now = time.time()
        for cid, kind, label in targets:
            if kind == "eacc":
                m, n = _mean_across(
                    engine, [f"eacc{i}" for i in range(1, 23)])
            elif kind == "pacc":
                m, n = _mean_across(
                    engine, [f"pacc{i}" for i in range(1, 23)])
            else:
                ring = kind if kind.startswith("facc_") else f"facc{cid}"
                m, n = _pool_mean(engine, ring)
            if n < MIN_N:
                continue
            act = self.active.get(cid)
            bact = self.boosted.get(cid)
            if bact is not None and act is None:
                # a boosted winner is re-judged every pass: fall back
                # under the OK bar and the earned weight is withdrawn
                # BEFORE any damp is considered; keep winning and the
                # boost deepens toward the cap, one step per cooldown.
                if m < REWARD_OK:
                    self.boost.pop(cid, None)
                    self.boosted.pop(cid, None)
                    rec = dict(bact)
                    rec.update(at=now, reward=round(m, 3), n=n,
                               action="boost withdrawn (reward faded)",
                               state="boost-withdrawn")
                    self.actions.append(rec)
                    taken.append(rec)
                elif n >= int(bact.get("n", 0)) + COMMIT_N and \
                        m > REWARD_HIGH and \
                        self.boost.get(cid, 1.0) < BOOST_CAP:
                    self.boost[cid] = round(min(
                        BOOST_CAP, self.boost.get(cid, 1.0) * BOOST_STEP), 3)
                    bact["n"] = n
                    bact["reward"] = round(m, 3)
                    bact["action"] = f"boost ×{self.boost[cid]:.2f}"
                    self.actions.append(dict(bact))
                    taken.append(dict(bact))
                continue
            if act is None and bact is None and m > REWARD_HIGH:
                # WINNER BOOST: sustained reward earns weight, capped.
                self.boost[cid] = BOOST_STEP
                rec = {"at": now, "kind": label, "id": int(cid),
                       "reward": round(m, 3), "n": n,
                       "action": f"boost ×{BOOST_STEP:.2f}",
                       "state": "boosted"}
                self.boosted[cid] = rec
                self.actions.append(rec)
                taken.append(rec)
                continue
            if act is None:
                if m < REWARD_LOW:
                    self.damp[cid] = max(DAMP_FLOOR, DAMP_STEP)
                    rec = {"at": now, "kind": label, "id": int(cid),
                           "reward": round(m, 3), "n": n,
                           "action": f"damp ×{self.damp[cid]:.2f}",
                           "state": "active"}
                    self.active[cid] = rec
                    self.actions.append(rec)
                    taken.append(rec)
            else:
                cooled = n >= int(act.get("n", 0)) + COMMIT_N
                if m >= REWARD_OK:
                    self.damp.pop(cid, None)
                    self.active.pop(cid, None)
                    rec = dict(act)
                    rec.update(at=now, reward=round(m, 3), n=n,
                               action="recovered → weight restored",
                               state="recovered")
                    self.actions.append(rec)
                    taken.append(rec)
                elif cooled and m < REWARD_LOW and \
                        self.damp.get(cid, 1.0) > DAMP_FLOOR:
                    self.damp[cid] = round(
                        max(DAMP_FLOOR, self.damp[cid] * DAMP_STEP), 3)
                    act["n"] = n
                    act["reward"] = round(m, 3)
                    act["action"] = f"deepen ×{self.damp[cid]:.2f}"
                    self.actions.append(dict(act))
                    taken.append(dict(act))
        if taken:
            self.actions = self.actions[-ACTION_CAP:]
            self.save()
        return taken

    # ------------------------------------------------ 2) issue bandit
    def issue_threshold(self, context: str | None = None) -> float:
        """Threshold for a context once that context has earned its own
        bucket; the global bucket otherwise (fresh = 0.35, legacy)."""
        if context:
            cb = self.contexts.get(context)
            if cb is not None:
                return float(BUCKETS[int(cb["bucket"])])
        return float(BUCKETS[self.bandit["bucket"]])

    def on_settle(self, issued: bool, won: bool | None,
                  context: str | None = None) -> None:
        """Score the committed bucket on ONE settled window:
        issued → ±1, no-trade →0 (opportunity-neutral). The global
        bandit always learns; a named context learns in parallel and
        takes over its own threshold once it has a committed bucket."""
        r = 0.0 if not issued else (1.0 if won else -1.0)
        self._score_bandit(self.bandit, r, "bandit")
        if context:
            cb = self.contexts.setdefault(context, self._fresh_bandit())
            self._score_bandit(cb, r, f"bandit:{context}")
        self.save()

    def _score_bandit(self, bandit: dict, r: float, kind: str) -> None:
        st = bandit["stats"].setdefault(
            str(bandit["bucket"]), {"n": 0, "reward": 0.0})
        st["n"] += 1
        st["reward"] = round(st["reward"] + r, 3)
        bandit["committed"] += 1
        if bandit["committed"] >= COMMIT_N:
            self._repick(bandit, kind)

    @staticmethod
    def _bandit_mean(bandit: dict, i: int) -> float:
        st = bandit["stats"].get(str(i)) or {"n": 0, "reward": 0.0}
        n = int(st["n"])
        return (float(st["reward"]) / n) if n > 0 else float("-inf")

    def _mean_reward(self, i: int) -> float:
        return self._bandit_mean(self.bandit, i)

    def _repick(self, bandit: dict | None = None,
                kind: str = "bandit") -> None:
        bandit = self.bandit if bandit is None else bandit
        best = max(range(len(BUCKETS)),
                   key=lambda i: (self._bandit_mean(bandit, i)
                                  if bandit["stats"].get(str(i), {})
                                  .get("n") else float("-inf")))
        if random.random() < EXPLORE:
            others = [i for i in range(len(BUCKETS)) if i != best]
            best = random.choice(others) if others else best
        if best != bandit["bucket"]:
            bandit["bucket"] = int(best)
            self.actions.append({
                "at": time.time(), "kind": kind, "id": 0,
                "action": f"issue threshold → {BUCKETS[best]:.2f}",
                "reward": round(self._bandit_mean(bandit, best), 3),
                "n": 0, "state": "repicked"})
        bandit["committed"] = 0

    # ------------------------------------------- 3) Q-table over states
    def choose_mix(self, state: str, key: str) -> str:
        """ε-greedy subsystem-mix action for a window state. A state
        with <Q_MIN_N decided samples plays 'balanced' (legacy mix) —
        the table never guesses a mix it has not measured."""
        cell = self.q.get(state) or {}
        known = {a: v for a, v in cell.items() if int(v.get("n", 0)) > 0}
        if sum(int(v["n"]) for v in known.values()) < Q_MIN_N:
            action = "balanced"
        elif random.random() < EXPLORE:
            action = random.choice(list(MIX_ACTIONS))
        else:
            action = max(known, key=lambda a: float(known[a]["q"])) \
                if known else "balanced"
        self._q_pending[key] = {"state": state, "action": action}
        return action

    @staticmethod
    def mix_shares(shares: dict[str, float], action: str) -> dict[str, float]:
        """Apply a Q-table mix action to subsystem shares (renormalised
        to 1.0 — same rule as every other share adjustment)."""
        mult = MIX_ACTIONS.get(action) or {}
        if not mult:
            return dict(shares)
        out = {k: float(v) * float(mult.get(k, 1.0))
               for k, v in shares.items()}
        total = sum(out.values()) or 1.0
        return {k: v / total for k, v in out.items()}

    def q_on_settle(self, state: str | None, action: str | None,
                    reward: float | None) -> None:
        """One-step Q update Q += α(r − Q) for the (state, action) pair
        frozen into the settled lock. Decided windows only — a no-trade
        window grades the bandit, never the mix."""
        if not state or not action or reward is None:
            return
        if action not in MIX_ACTIONS:
            return
        cell = self.q.setdefault(state, {})
        cur = cell.setdefault(action, {"q": 0.0, "n": 0})
        cur["q"] = round(float(cur["q"])
                         + Q_ALPHA * (float(reward) - float(cur["q"])), 4)
        cur["n"] = int(cur["n"]) + 1
        self.save()

    # ------------------------------------------- 3) honest target tracker
    def target_status(self, ledger: Any) -> dict:
        wins = losses = 0
        rows = getattr(ledger, "rows", {}) or {}
        for _k, rs in rows.items():
            for r in rs:
                if r.get("result") == "win":
                    wins += 1
                elif r.get("result") == "loss":
                    losses += 1
        n = wins + losses
        cur = round(wins / n, 4) if n else None
        gap = (round((ACC_TARGET - cur) * 100, 1)
               if cur is not None else None)
        if n == 0:
            status = "no settled windows yet"
        elif cur is not None and cur >= ACC_TARGET:
            status = "TARGET MET — keep proving it (never relax)"
        elif n < 40:
            status = f"preliminary — needs n≥40 to speak (have {n})"
        else:
            status = ("TARGET NOT MET — RL climbing with the data; "
                      f"gap {gap} pp, measured honestly")
        return {"target": ACC_TARGET, "current": cur, "n": n,
                "wins": wins, "losses": losses, "gap_pp": gap,
                "status": status}

    # ---------------------------------------------------------- console
    def status(self) -> dict:
        return {
            "version": self.version,
            "threshold": self.issue_threshold(),
            "bandit": {
                "bucket": self.bandit["bucket"],
                "committed": self.bandit["committed"],
                "commit_n": COMMIT_N,
                "stats": {BUCKETS[int(i)]: v
                          for i, v in self.bandit["stats"].items()},
            },
            "damp": {k: v for k, v in sorted(self.damp.items())},
            "boost": {k: v for k, v in sorted(self.boost.items())},
            "factors": {str(k): round(self.factor(int(k)), 3) for k in
                        sorted(set(self.damp) | set(self.boost))},
            "active_fixes": list(self.active.values()),
            "active_boosts": list(self.boosted.values()),
            "contexts": {
                ctx: {"threshold": float(BUCKETS[int(b["bucket"])]),
                      "committed": b["committed"],
                      "stats": {BUCKETS[int(i)]: v
                                for i, v in b["stats"].items()}}
                for ctx, b in sorted(self.contexts.items())},
            "qtable": {
                state: {a: {"q": round(float(v["q"]), 3),
                            "n": int(v["n"])}
                        for a, v in sorted(cell.items())}
                for state, cell in sorted(self.q.items())},
            "mix_actions": list(MIX_ACTIONS),
            "recent_actions": self.actions[-8:][::-1],
        }


_RL: RLSystem | None = None


def get_rl() -> RLSystem:
    global _RL
    if _RL is None:
        _RL = RLSystem()
    return _RL
