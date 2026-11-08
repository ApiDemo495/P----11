"""Vote-change lab (user rule: CHANGE components, never remove).

Every captured fusion component keeps its vote. What changes is
each voter's WEIGHT, fitted from its own measured barrier EV:
- positive EV  -> positive weight (vote as-is)
- negative EV  -> NEGATIVE weight (the vote is flipped — changed,
  not removed: a consistently-wrong signal still contributes,
  in the direction the data says it should)
- near-zero EV -> minimum weight 0.005 (nobody is zeroed out)
A trade needs >= TAU of total |weight| agreeing. Weights are
fitted on a train slice and judged on untouched data, in both
directions (fit fwd -> test back, fit back -> test fwd).
"""
from __future__ import annotations

from .experiment import _m
from .tp4 import settle_r, split, _sign

FLOOR = 0.005
CLIP = 0.3


def fit_changed_weights(part, names):
    w = {}
    for n in names:
        vals = [settle_r(r, _sign((r.get("comp") or {}).get(n, 0.0)),
                         0.75, 2.0) for r in part]
        vals = [x for x in vals if x is not None]
        ev = _m(vals)["ev_r"] or 0.0
        v = max(-CLIP, min(CLIP, ev))
        if abs(v) < FLOOR:
            v = FLOOR if v >= 0 else -FLOOR
        w[n] = round(v, 4)
    return w


def changed_vote(comp, weights, tau=0.5):
    """(direction, agreement). Every name in weights votes."""
    tot = sum(abs(v) for v in weights.values()) or 1.0
    s = sum(w * _sign(comp.get(n, 0.0)) for n, w in weights.items())
    frac = abs(s) / tot
    return ((1 if s > 0 else -1 if s < 0 else 0) if frac >= tau else 0), frac


def run_vote(part, weights, tau=0.5, rsi=True, ratio=2.0):
    from .tp4 import rule_dir
    vals = []
    for r in part:
        d = rule_dir(r, "rsi_revert") if rsi else 0
        if d == 0:
            d, _ = changed_vote(r.get("comp") or {}, weights, tau)
        v = settle_r(r, d, 0.75, ratio)
        if v is not None:
            vals.append(v)
    return vals


def evaluate_both(rows, tau=0.5):
    names = sorted({n for r in rows for n in (r.get("comp") or {})})
    train, test = split(rows)
    w_f = fit_changed_weights(train, names)
    w_r = fit_changed_weights(test, names)
    fwd = _m(run_vote(test, w_f, tau))
    rev = _m(run_vote(train, w_r, tau))
    pooled = _m(run_vote(test, w_f, tau) + run_vote(train, w_r, tau))
    return {"weights_forward_fit": w_f, "forward_test": fwd,
            "reverse_test": rev, "pooled_oos": pooled}
