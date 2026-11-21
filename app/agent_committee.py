"""§5 — the four-agent committee: four always-active, keyless
sub-agents that vote on EVERY prediction.

    1. FlowAgent          — microstructure / order-flow
    2. LeadLagAgent       — cross-asset lead-lag
    3. ContrarianAgent    — behavioural contrarian (carries the VETO)
    4. GovernorAgent      — meta-cognition risk governor (carries the
                            SHADOW / risk-throttle VETO)

No API keys, no model downloads — every agent is pure measurement over
the market bundle, the emotion pass, the regime engine and the
metacognition book, so a prediction always has four independent reads.

An optional async LLM overlay can be attached when keys ARE configured
(`set_llm`); without keys the committee is fully local and the overlay
simply never runs.  The overlay never replaces a vote — it annotates.

Issue rules (directive §5 + §0 honesty):
    * consensus direction must agree with the signal's side;
    * no veto → `may_issue` False (the prediction ships as WATCH);
    * Governor vetoes SHADOW cells and risk-throttled regimes — no
      capital on an unconfirmed cell, whatever the other three say.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class AgentVote:
    agent: str
    direction: float          # [-1, +1]: sign = side, |·| = conviction
    confidence: float         # [0, 1]
    reason: str
    veto: bool = False
    veto_reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CommitteeResult:
    votes: list[AgentVote]
    consensus: float          # weighted mean direction [-1, +1]
    agreement: float          # 0..1 — how unanimous the board is
    may_issue: bool
    vetoes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    llm_note: str | None = None

    def as_dict(self) -> dict:
        return {
            "votes": [v.as_dict() for v in self.votes],
            "consensus": round(self.consensus, 4),
            "agreement": round(self.agreement, 4),
            "may_issue": self.may_issue,
            "vetoes": list(self.vetoes),
            "notes": list(self.notes),
            "llm_note": self.llm_note,
        }


# ══════════════════════════════════════════════════════════════════════
# 1 · microstructure / order-flow
# ══════════════════════════════════════════════════════════════════════
def flow_vote(market: dict) -> AgentVote:
    """OFI/CMP/flow-direction read of the tape."""
    ofti = float(market.get("ofti") or 0.0)
    cmp_ = float(market.get("cmp_divergence") or market.get("cmp") or 0.0)
    spread = float(market.get("spread_bps") or 0.0)
    closes = list(market.get("closes") or [])
    volumes = list(market.get("volumes") or [])

    # last-minute return as the flow's immediate expression
    mom = 0.0
    if len(closes) >= 2 and closes[-2]:
        mom = (closes[-1] - closes[-2]) / closes[-2]
    # volume climax in the flow's direction
    climax_dir = 0.0
    if len(volumes) >= 6 and sum(volumes[-30:] or [0]) > 0:
        v5 = sum(volumes[-5:]) / 5.0
        v30 = sum(volumes[-min(30, len(volumes)):]) / min(30, len(volumes))
        if v30 > 0:
            climax_dir = _clamp((v5 / v30 - 1.0), -1.0, 1.0) * (
                1.0 if mom >= 0 else -1.0)

    raw = _clamp(0.45 * _clamp(ofti * 2.5, -1, 1)
                 + 0.30 * _clamp(cmp_ * 2.0, -1, 1)
                 + 0.25 * climax_dir, -1.0, 1.0)
    # thin/illiquid tape (wide spread) → low conviction, never a veto
    conf = _clamp(0.55 + 0.30 * abs(raw) - min(spread, 30.0) / 60.0, 0.05, 0.95)
    reason = (f"ofti {ofti:+.2f}, cmp {cmp_:+.2f}, climax "
              f"{climax_dir:+.2f}, spread {spread:.1f}bp")
    return AgentVote("flow", raw, conf, reason)


# ══════════════════════════════════════════════════════════════════════
# 2 · cross-asset lead-lag
# ══════════════════════════════════════════════════════════════════════
def lead_lag_vote(market: dict, symbol: str = "BTC") -> AgentVote:
    """Leader → follower read: the leader's most recent return is the
    follower's expected move; for the leader itself, use the peer mean
    as the 'market beta' confirmation."""
    peers: list[list[float]] = [
        list(r) for r in (market.get("peer_returns") or [])
        if isinstance(r, (list, tuple)) and len(r) >= 2
    ]
    closes = list(market.get("closes") or [])
    if len(peers) < 6 or len(closes) < 6:
        return AgentVote("lead_lag", 0.0, 0.10,
                         "insufficient cross-section for lead-lag",
                         veto=False)

    # peer mean return this minute vs its own lag — the 'market' move
    series = [_mean([float(x) for x in row]) for row in peers]
    # own last return
    own_now = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] else 0.0
    own_prev = (closes[-2] - closes[-3]) / closes[-3] if (
        len(closes) >= 3 and closes[-3]) else 0.0
    peer_now = series[-1]
    peer_prev = series[-2] if len(series) >= 2 else 0.0

    if symbol == "BTC":
        # BTC is the leader: confirm own continuation with the crowd's
        # agreement (peers following the same direction)
        agree = 1.0 if (own_now >= 0) == (peer_now >= 0) else -1.0
        raw = _clamp(agree * min(1.0, abs(own_now) * 200.0 + 0.3), -1, 1)
        conf = 0.45 + 0.25 * (1.0 if agree > 0 else 0.0)
        reason = f"leader continuation, peers {'confirm' if agree > 0 else 'diverge'}"
    else:
        # follower: leader's LAST return predicts our NEXT move
        lead = peer_prev
        raw = _clamp(lead * 300.0, -1.0, 1.0)
        # and our own last move already absorbed some of it — dampen
        raw *= _clamp(1.0 - abs(own_prev) * 100.0, 0.3, 1.0)
        conf = _clamp(0.35 + 0.30 * abs(raw), 0.05, 0.85)
        reason = (f"leader lag {lead:+.4f} → expected follow "
                  f"(own absorbed {own_now:+.4f})")
    return AgentVote("lead_lag", raw, conf, reason)


# ══════════════════════════════════════════════════════════════════════
# 3 · behavioural contrarian (VETO carrier #1)
# ══════════════════════════════════════════════════════════════════════
def contrarian_vote(emotion: dict | None) -> AgentVote:
    """Fade the crowd at extremes; VETO when the crowd-vs-smart split
    warning says retail is chasing while smart money distributes."""
    if not emotion:
        return AgentVote("contrarian", 0.0, 0.10, "no emotion pass yet")

    cerc_vec = emotion.get("cerc_vectors") or {}
    retail = float(cerc_vec.get("retail_herd") or 0.0)
    smart = float(cerc_vec.get("smart_money") or 0.0)
    split = float(cerc_vec.get("crowd_smart_split") or 0.0)
    emos = emotion.get("emotions") or {}

    # chase-fatigue / FOMO extremes: the more extended the crowd, the
    # stronger the fade direction
    chase = float(emos.get(18) or 0.5) - 0.5     # + = crowd chasing up
    fomo = float(emos.get(20) or 0.0)            # event magnitude
    panic = float(emos.get(1) or 0.0)            # event magnitude

    fade = _clamp(-2.0 * chase - 0.5 * math.copysign(fomo, chase or 1.0)
                  + 0.6 * (0.5 - retail) * 2.0, -1.0, 1.0)
    # panic high → contrarian BIAS up (buy fear)
    if panic > 0.5:
        fade = _clamp(fade + 0.4, -1.0, 1.0)

    conf = _clamp(0.40 + 0.35 * abs(fade), 0.05, 0.9)
    reason = (f"retail_herd {retail:+.2f}, smart {smart:+.2f}, "
              f"chase {chase:+.2f}, panic {panic:.2f}")

    # VETO: high-conviction contrarian warning — retail euphoric while
    # smart money distributes.  The prediction does not issue into it.
    if split >= 0.6:
        return AgentVote("contrarian", fade, conf, reason, veto=True,
                         veto_reason=(f"crowd-vs-smart split {split:.2f} ≥ 0.60 "
                                      f"— retail chasing while smart money "
                                      f"distributes"))
    return AgentVote("contrarian", fade, conf, reason)


# ══════════════════════════════════════════════════════════════════════
# 4 · meta-cognition risk governor (VETO carrier #2)
# ══════════════════════════════════════════════════════════════════════
def governor_vote(metacog_status: str, pillar_report: Any,
                  regime: dict | None, allows_capital: bool = True
                  ) -> AgentVote:
    """The governor reads the AUDIT, not the tape: SHADOW cells and
    risk-throttled regimes veto issuance outright."""
    notes: list[str] = []
    conf = 0.7
    direction = 0.0
    veto = False
    veto_reason = ""

    if metacog_status != "CONFIRMED":
        veto = True
        veto_reason = (f"cell is {metacog_status} — unconfirmed cells run "
                       f"no capital (directive §4)")
        notes.append(veto_reason)
    if pillar_report is not None and getattr(pillar_report, "abstain", False):
        veto = True
        veto_reason = (veto_reason + "; " if veto_reason else "") + \
            "pillar audit abstain: " + ", ".join(
                getattr(pillar_report, "reasons", [])[:2])
        notes.append("pillar audit abstain")
    if regime:
        if regime.get("risk_throttle"):
            veto = True
            veto_reason = (veto_reason + "; " if veto_reason else "") + \
                f"regime {regime.get('state')} throttles risk"
            notes.append("regime risk throttle")
        # regime tilt nudges the governor's direction read
        conf = _clamp(float(regime.get("confidence") or 0.5), 0.1, 0.95)
        state = str(regime.get("state") or "")
        if state.endswith("BULL"):
            direction = 0.3
        elif state.endswith("BEAR"):
            direction = -0.3
    if not allows_capital and not veto:
        veto = True
        veto_reason = "capital not allowed on this cell"
        notes.append(veto_reason)

    reason = "; ".join(notes) if notes else (
        f"cell {metacog_status}, regime conf {conf:.2f} — clear to vote")
    return AgentVote("governor", direction, conf, reason,
                     veto=veto, veto_reason=veto_reason)


# ══════════════════════════════════════════════════════════════════════
# committee façade
# ══════════════════════════════════════════════════════════════════════
#: weights — the governor's direction is weak by design (it gates, it
#: does not steer); flow and lead-lag carry the side; contrarian fades.
WEIGHTS = {"flow": 0.32, "lead_lag": 0.28, "contrarian": 0.22,
           "governor": 0.18}


class AgentCommittee:
    """Runs the four agents on every prediction and merges their votes.

    Optional async LLM overlay: attach with `set_llm(fn)` where fn is
    an async callable(prompt) -> str.  Without keys the overlay stays
    None and nothing external is called.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._llm: Callable[..., Any] | None = None
        self.last: CommitteeResult | None = None
        self.vote_count = 0

    def set_llm(self, fn: Callable[..., Any] | None) -> None:
        with self._lock:
            self._llm = fn

    # ------------------------------------------------------------- vote
    def vote(self, market: dict | None,
             emotion: dict | None,
             regime: dict | None,
             metacog_status: str = "SHADOW",
             pillar_report: Any = None,
             allows_capital: bool = False,
             symbol: str = "BTC") -> CommitteeResult:
        market = market or {}
        votes = [
            flow_vote(market),
            lead_lag_vote(market, symbol=symbol),
            contrarian_vote(emotion),
            governor_vote(metacog_status, pillar_report, regime,
                          allows_capital=allows_capital),
        ]
        by_name = {v.agent: v for v in votes}
        total_w = sum(WEIGHTS[v.agent] for v in votes) or 1.0
        consensus = sum(WEIGHTS[v.agent] * v.direction
                        for v in votes) / total_w
        # agreement: 1 − (spread of signed directions around consensus)
        spread = _mean([abs(v.direction - consensus) for v in votes])
        agreement = _clamp(1.0 - spread, 0.0, 1.0)

        vetoes = [v.veto_reason or f"{v.agent} veto"
                  for v in votes if v.veto and v.veto_reason]
        notes: list[str] = []
        may_issue = not vetoes
        if may_issue and abs(consensus) < 0.12:
            may_issue = False
            notes.append(f"consensus {consensus:+.2f} inside the ±0.12 "
                         f"dead band — board too quiet to issue")
        if may_issue and agreement < 0.35:
            may_issue = False
            notes.append(f"board agreement {agreement:.2f} < 0.35 — "
                         f"agents divided, ship as WATCH")
        for v in votes:
            if v.veto and v.veto_reason and v.veto_reason not in vetoes:
                vetoes.append(v.veto_reason)

        result = CommitteeResult(
            votes=votes, consensus=consensus, agreement=agreement,
            may_issue=may_issue, vetoes=vetoes, notes=notes)
        with self._lock:
            self.last = result
            self.vote_count += 1
        return result

    # ------------------------------------------------------- LLM overlay
    async def annotate(self, result: CommitteeResult,
                       prompt_ctx: str = "") -> CommitteeResult:
        """Optional: when an LLM is configured, ask for a one-line
        overlay note.  Never changes votes or `may_issue`."""
        with self._lock:
            fn = self._llm
        if fn is None:
            return result
        try:
            prompt = (
                "Four-agent trading committee summary (do not trade, "
                "annotate only):\n" + prompt_ctx + "\n"
                f"consensus={result.consensus:+.2f} "
                f"agreement={result.agreement:.2f} "
                f"vetoes={result.vetoes}\n"
                "Reply with one sentence of risk commentary.")
            note = await fn(prompt)
            result.llm_note = str(note)[:300]
        except Exception:
            result.llm_note = None
        return result


#: process-wide singleton — every prediction votes through this board
COMMITTEE = AgentCommittee()
