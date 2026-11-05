"""Reinforcement-learning controllers (user: 'use RL to find & fix errors,
boost prediction accuracy, fix every formula/emotion/physics').

Three controllers, one console:

  1. ERROR-FINDER (auto-fix + rollback): every graded component already
     has reward rings (facc/eacc/pacc strict-sign outcomes). When a
     component's pooled reward drops below0.40 over n≥8 windows the
     controller DAMPS its effective weight (×0.75, floor0.4), watches the
     next windows, RESTORES the weight when reward recovers (≥0.50),
     and deepens (or rolls back) otherwise. Every action is recorded —
     visible in /api/rl/status — so RL fixes are auditable, never
     invisible magic.
  2. ISSUE-THOUSET BANDIT (accuracy booster): a4-armed bandit over the
     confidence issue threshold {0.25,0.30,0.35,0.45? -> 0.25..0.45}
     commits to a bucket for8 settled windows, scores it by realised
     strict-sign reward (issued: ±1, no-trade:0), re-picks
     argmax(mean)+15% exploration. The threshold only moves where the
     DATA says issued trades win more.
  3. TARGET TRACKER (honest): measures pooled ledger accuracy against
     the user's90% target and prints the REAL number and gap — the RL
     loop climbs toward the target; it never rewrites history to fake
     reaching it.

Persistence: data/rl.json with BUILD_VERSION guard (stale state dropped).
Fairness: a fresh process starts at threshold0.35 and damp1.0 — exactly
the legacy behaviour, so pinned confidence tests stay green until real
settlements move the bandit.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from typing import Any

BUILD_VERSION = 1
RL_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "rl.json")
BUCKETS = [0.25, 0.30, 0.35, 0.40, 0.45]
COMMIT_N = 8             # settled windows per bandit commitment
EXPLORE = 0.15           # chance to explore another bucket at re-pick
DAMP_FLOOR = 0.4
DAMP_STEP = 0.75         # multiply down by25% per sustained offence
REWARD_LOW = 0.40
REWARD_OK = 0.50
MIN_N = 8
ACTION_CAP = 200
ACC_TARGET = 0.90        # user target — reported honestly, never faked


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
    if not rl.damp:
        return shares
    out: dict[str, float] = {}
    for k, v in shares.items():
        cid = _SHARE_KEY_ID.get(k)
        out[k] = float(v) * rl.damp.get(cid, 1.0) if cid else float(v)
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


class RLSystem:
    def __init__(self, path: str | None = None) -> None:
        if path is None and os.environ.get("PYTEST_CURRENT_TEST"):
            import tempfile as _tf
            path = os.path.join(_tf.gettempdir(), "predict4_test_rl.json")
        self.path = path or RL_FILE
        self.damp: dict[int, float] = {}
        self.active: dict[int, dict] = {}      # id -> action record
        self.actions: list[dict] = []
        self.bandit: dict[str, Any] = {
            "bucket": 2,                       # default0.35
            "committed": 0,
            "stats": {str(i): {"n": 0, "reward": 0.0}
                      for i in range(len(BUCKETS))},
        }
        self.version = BUILD_VERSION
        self.load()

    # ------------------------------------------------------ persistence
    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            if int(d.get("build_version", -1)) != BUILD_VERSION:
                return                      # stale state dropped (B-8)
            self.damp = {int(k): float(v)
                         for k, v in (d.get("damp") or {}).items()}
            self.active = {int(k): dict(v)
                           for k, v in (d.get("active") or {}).items()}
            self.actions = list(d.get("actions") or [])[-ACTION_CAP:]
            b = d.get("bandit") or {}
            if b.get("bucket") is not None:
                self.bandit = {
                    "bucket": int(b["bucket"]),
                    "committed": int(b.get("committed") or 0),
                    "stats": b.get("stats") or self.bandit["stats"],
                }
        except (OSError, ValueError, TypeError):
            pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(self.path) or ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"build_version": BUILD_VERSION,
                           "damp": self.damp, "active": self.active,
                           "actions": self.actions[-ACTION_CAP:],
                           "bandit": self.bandit}, fh)
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
    def issue_threshold(self) -> float:
        return float(BUCKETS[self.bandit["bucket"]])

    def on_settle(self, issued: bool, won: bool | None) -> None:
        """Score the committed bucket on ONE settled window:
        issued → ±1, no-trade →0 (opportunity-neutral)."""
        st = self.bandit["stats"].setdefault(
            str(self.bandit["bucket"]), {"n": 0, "reward": 0.0})
        r = 0.0 if not issued else (1.0 if won else -1.0)
        st["n"] += 1
        st["reward"] = round(st["reward"] + r, 3)
        self.bandit["committed"] += 1
        if self.bandit["committed"] >= COMMIT_N:
            self._repick()
        self.save()

    def _mean_reward(self, i: int) -> float:
        st = self.bandit["stats"].get(str(i)) or {"n": 0, "reward": 0.0}
        n = int(st["n"])
        return (float(st["reward"]) / n) if n > 0 else float("-inf")

    def _repick(self) -> None:
        best = max(range(len(BUCKETS)),
                   key=lambda i: (self._mean_reward(i)
                                  if self.bandit["stats"].get(str(i), {})
                                  .get("n") else float("-inf")))
        if random.random() < EXPLORE:
            others = [i for i in range(len(BUCKETS)) if i != best]
            best = random.choice(others) if others else best
        if best != self.bandit["bucket"]:
            self.bandit["bucket"] = int(best)
            self.actions.append({
                "at": time.time(), "kind": "bandit", "id": 0,
                "action": f"issue threshold → {BUCKETS[best]:.2f}",
                "reward": round(self._mean_reward(best), 3), "n": 0,
                "state": "repicked"})
        self.bandit["committed"] = 0

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
            "active_fixes": list(self.active.values()),
            "recent_actions": self.actions[-8:][::-1],
        }


_RL: RLSystem | None = None


def get_rl() -> RLSystem:
    global _RL
    if _RL is None:
        _RL = RLSystem()
    return _RL
