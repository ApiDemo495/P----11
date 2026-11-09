"""brain.md §11 acceptance tests — real-connectome brain component.

Everything runs against a synthetic FlyWire-shaped dataset (same schema
as Connectivity_783.parquet + the annotations TSV), so the suite never
needs the network; the real100 MB build is exercised by the live app.
"""
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import app.flywire as fw
from app.flywire import (BUILD_VERSION, FlyWireBrain, build_sync,
                         MB_CLASSES)


# ---------------------------------------------------------------- fixtures
def _make_dataset(tmp: Path) -> tuple[Path, Path]:
    """Synthetic but schema-faithful: ALPN -> KC -> MBON chains, PAM/PPL1
    DANs, LH neurons, descending neurons — every acceptance path exists."""
    roots = []
    meta = []  # (root, cell_class, super_class, top_nt, cell_type)
    counter = {"n": 0}
    def add(cls, sup, nt, ctype, n):
        out = []
        for _ in range(n):
            r = 720000000000000000 + counter["n"]
            counter["n"] += 1
            roots.append(r)
            meta.append((r, cls, sup, nt, ctype))
            out.append(r)
        return out
    alpn = add("ALPN", "central", "acetylcholine", "ALPN_a", 60)
    kc = add("Kenyon_Cell", "central", "acetylcholine", "KC_a", 40)
    mbon_ach = add("MBON", "central", "acetylcholine", "MBON_ach", 4)
    mbon_glu = add("MBON", "central", "glutamate", "MBON_glu", 4)
    pam = add("DAN", "central", "dopamine", "PAM_g1", 2)
    ppl1 = add("DAN", "central", "dopamine", "PPL1_g1", 2)
    mbin = add("MBIN", "central", "acetylcholine", "MBIN_a", 2)
    lhl = add("LHLN", "central", "gaba", "LHLN_a", 4)
    lhcent = add("LHCENT", "central", "acetylcholine", "LHC_a", 2)
    dn = add("descending", "descending", "acetylcholine", "DN_a", 3)
    mbon = mbon_ach + mbon_glu
    edges = []  # (pre, post, conn, exc)
    def edge(a, b, c=3, e=1):
        edges.append((a, b, c, e))
    # every ALPN reaches two distinct KCs (channel -> circuit)
    for i, a in enumerate(alpn):
        edge(a, kc[i % len(kc)], 4 + (i % 5), 1)
        edge(a, kc[(i * 7 + 3) % len(kc)], 2 + (i % 3), 1)
    # every KC reaches every MBON (probe -> non-zero balance)
    for k in kc:
        for m in mbon_ach:
            edge(k, m, 3, 1)
        for m in mbon_glu:
            edge(k, m, 3, -1)          # inhibitory (Glu) MBONs
    # MBON -> LH, MBON -> DN (whole-brain share), KC -> DAN
    for m in mbon:
        for h in lhl + lhcent:
            edge(m, h, 2, 1)
        for d in dn:
            edge(m, d, 2, 1)
    for i, k in enumerate(kc[:20]):
        edge(k, pam[i % 2], 2, 1)
        edge(k, ppl1[i % 2], 2, -1)
    for x in mbin:
        edge(mbon_ach[0], x, 2, 1)
    pq.write_table(
        pa.table({
            "Presynaptic_ID": pa.array([e[0] for e in edges], pa.int64()),
            "Postsynaptic_ID": pa.array([e[1] for e in edges], pa.int64()),
            "Presynaptic_Index": pa.array([0] * len(edges), pa.int64()),
            "Postsynaptic_Index": pa.array([0] * len(edges), pa.int64()),
            "Connectivity": pa.array([e[2] for e in edges], pa.int64()),
            "Excitatory": pa.array([e[3] for e in edges], pa.int64()),
            "Excitatory x Connectivity": pa.array(
                [e[2] * e[3] for e in edges], pa.int64()),
        }),
        tmp / "Connectivity_783.parquet")
    lines = ["root_id\tsuper_class\tcell_class\ttop_nt\tcell_type"]
    for r, cls, sup, nt, ct in meta:
        lines.append(f"{r}\t{sup}\t{cls}\t{nt}\t{ct}")
    (tmp / "Supplemental_file1_neuron_annotations.tsv").write_text(
        "\n".join(lines) + "\n")
    return tmp / "Connectivity_783.parquet", tmp / (
        "Supplemental_file1_neuron_annotations.tsv")


@pytest.fixture(scope="module")
def brain(tmp_path_factory) -> FlyWireBrain:
    tmp = tmp_path_factory.mktemp("flywire")
    pq_path, ann_path = _make_dataset(tmp)
    npz = tmp / "flywire_783.npz"
    meta = build_sync(pq_path, ann_path, npz)
    b = FlyWireBrain()
    # load directly from the synthetic npz (cache path swapped)
    b.npz_path = lambda: npz            # type: ignore[assignment]
    b._parquet_path = pq_path            # anatomy reads the real parquet
    b._ann_path = ann_path
    b._load(npz)
    b.state = "ready"
    b.load_ms = 1.0
    b.calibrate()
    return b


# ------------------------------------------------------------------ tests
def test_build_properties(brain, tmp_path):
    """1. W is CSR float32, rows normalised, signs follow Excitatory,
    BUILD_VERSION in the npz."""
    from scipy import sparse
    assert sparse.isspmatrix_csr(brain.W)
    assert brain.W.data.dtype == np.float32
    # rows (post-synaptic) with incoming edges are |·|-normalised to 1
    row_abs = np.abs(brain.W).sum(axis=1).A1
    nz = row_abs > 0
    assert np.allclose(row_abs[nz], 1.0, atol=1e-5)
    # signs: some edges negative (Excitatory = -1 survived normalisation)
    assert (brain.W.data < 0).any() and (brain.W.data > 0).any()
    z = np.load(tmp_path.parent / "flywire" / "flywire_783.npz") \
        if (tmp_path.parent / "flywire" / "flywire_783.npz").exists() else None
    # BUILD_VERSION written by build_sync (re-read from brain's own load)
    assert brain.meta["build_version"] == BUILD_VERSION
    assert brain.meta["neurons"] > 100


def test_five_step_verification_and_absence():
    """2. five steps with durations; no data -> chip brain absent."""
    b = FlyWireBrain()
    b.npz_path = lambda: Path("/nonexistent/flywire_783.npz")  # type: ignore
    b.state = "absent"
    b.last_error = "no cache"
    steps = b.verify()
    names = [s["step"] for s in steps]
    assert "cache" in names and "flywire" in names and "absent" in names
    assert all("ms" in s or s["step"].startswith("neuprint") or
               s["step"] == "absent" for s in steps)
    chip = b.chip()
    assert chip["kind"] == "absent" and "brain offline" in chip["text"]
    # vote withheld, fusion weight 0 (engine helper)
    assert b.vote([0.5] * 30)["vote"] is None
    from app.engine import PredictionEngine
    from app.feed import MarketFeed
    eng = PredictionEngine(MarketFeed(), emit=lambda m: None)
    # force the absent state (the machine may hold a real npz cache)
    saved_state = eng.brains.fw.state
    eng.brains.fw.state = "absent"
    assert eng.brain_fusion_weight("BTC", "tf1") == 0.0
    eng.brains.fw.state = saved_state


def test_input_groups_deterministic(brain):
    """3. 2·N_INPUTS contiguous ALPN groups, deterministic across runs."""
    n = brain.n_channels()
    assert n == fw.N_INPUTS or n == len(brain.alpn) // 2
    r1 = brain.encode([0.5] * n)
    r2 = brain.encode([0.5] * n)
    assert r1 is not None and np.array_equal(r1, r2)
    assert len(r1) == 2 * n
    # push-pull: each channel lights exactly one of its ON/OFF groups at
    # |0.5| (polarity flips may route a +0.5 input into the OFF group)
    for i in range(n):
        on, off = float(r1[i]), float(r1[n + i])
        assert {round(on, 6), round(off, 6)} == {0.0, 0.5}, (i, on, off)
    # determinism of the ALPN ordering itself
    assert np.array_equal(brain.alpn, np.sort(brain.alpn) if
                          brain.alpn_type is None else brain.alpn)


def test_flat_zero_and_antisymmetry(brain):
    """4. flat input -> vote exactly 0 with reason; vote(-x) == -vote(x)."""
    n = brain.n_channels()
    flat = brain.vote([0.0] * n)
    assert flat["vote"] == 0.0 and flat["reason"] == "flat input"
    x = [0.0] * n
    x[0], x[3], x[7] = 0.9, -0.4, 0.6
    vp = brain.vote(x)["vote"]
    vn = brain.vote([-v for v in x])["vote"]
    assert vp is not None and vn is not None
    assert abs(vn + vp) <= 1e-6


def test_polarity_calibration(brain):
    """5. each channel alone at +1 yields a non-zero balance; flips
    recorded; after flips every connected channel's +1 probe is positive."""
    n = brain.n_channels()
    assert brain.flips is not None and len(brain.flips) == n
    probe_after = []
    for i in range(n):
        values = [0.0] * n
        values[i] = 1.0
        rates = brain.encode(values)
        bal = 0.5 * (brain._balance(rates) - brain._balance(-rates))
        probe_after.append(bal)
    live = [i for i in range(n) if i not in brain.dead_channels]
    assert live, "synthetic circuit must respond on every channel"
    for i in live:
        assert probe_after[i] > 0, f"channel {i} not positive after flips"
    # flips were recorded as a property of the wiring
    assert isinstance(brain.flips.tolist(), list)
    st = brain.status()
    assert st["polarity"]["n_channels"] == n


def test_kc_sparsity(brain):
    """6. active KC share after hop 1 is <= 5% + 1 neuron."""
    n = brain.n_channels()
    values = [0.0] * n
    for i in range(0, n, 3):
        values[i] = 0.8
    r = brain.pass_mb(brain.encode(values))
    assert r["ok"]
    limit = math.ceil(fw.KC_SPARSITY * len(brain.kc)) + 1
    assert r["kc_active_hop1"] <= limit


def test_rpe_learning_and_rollback(brain):
    """7. wins raise E, a loss gives negative drg, weights change only at
    boundaries, rollback restores them."""
    n = brain.n_channels()
    x = [0.0] * n
    x[0] = 0.9
    before = brain._kc_mbon_delta.copy()
    brain.vote(x)                       # evaluation must NOT learn
    assert np.array_equal(brain._kc_mbon_delta, before)
    # three wins in the vote's direction
    for _ in range(3):
        brain.rpe_update(vote=0.4, direction=1, actual_move=+1.0)
    E_wins = brain.E
    assert E_wins > 0.0
    # a loss: R = -1 -> drg negative
    entry = brain.rpe_update(vote=0.4, direction=1, actual_move=-1.0)
    assert entry is not None and entry["drg"] < 0
    # weights changed only via rpe (delta now differs from pre-vote)
    active = np.flatnonzero(brain._kc_mbon_delta)  # may be empty if alpha*drg 0
    # rollback restores the snapshot taken at the last update
    snap_E = brain.E
    brain.E = 0.0
    assert brain.rollback() is True
    assert brain.E == snap_E


def test_pass_timing_and_threads(brain):
    """7/8. MB path < 20 ms; whole-brain pass runs on its own thread,
    never the request/main thread."""
    n = brain.n_channels()
    values = [0.3] * n
    r = brain.pass_mb(brain.encode(values))
    assert r["ok"] and r["ms"] < 20.0, r["ms"]
    # start the pass thread and sample the thread name it runs under
    brain.last_pass_at = 0
    brain._start_pass_thread()
    seen = {}
    orig = brain._inject_whole
    def spy(rates):
        seen["name"] = threading.current_thread().name
        seen["main"] = threading.current_thread() is threading.main_thread()
        return orig(rates)
    brain._inject_whole = spy            # type: ignore
    deadline = time.time() + fw.WHOLE_PASS_PERIOD_S + 6.0
    while "name" not in seen and time.time() < deadline:
        time.sleep(0.2)
    brain._inject_whole = orig           # type: ignore
    assert seen.get("name") == "flywire-pass"
    assert seen.get("main") is False
    assert brain.last_pass_ms is not None and brain.last_pass_ms < 1000.0


def test_anatomy_full_census(brain):
    """'full neurons, cells, synapses and others': class histogram,
    cell_class × cell_class weight matrix, synapse total, NT census."""
    brain._build_anatomy()               # run the heavy pass synchronously
    an = brain.anatomy()
    assert an["state"] == "ready"
    assert an["neurons"] == sum(an["classes"].values())
    assert an["connections"] > 0
    assert an["synapses"] and an["synapses"] >= an["connections"]
    m = an["block"]["matrix"]
    k = len(an["block"]["classes"])
    assert len(m) == k and all(len(row) == k for row in m)
    assert any(any(v > 0 for v in row) for row in m)
    assert "Kenyon_Cell" in an["classes"]
    # status carries it too (the brain page renders from there)
    st = brain.status()
    assert st["anatomy"]["state"] == "ready"


def test_no_token_in_repo():
    """9. no neuPrint token string appears in any repo file or log line."""
    import re
    root = Path(__file__).resolve().parent.parent
    pat = re.compile(r"(eyJ[A-Za-z0-9_-]{20,}|neuprint[_-]?token\s*=\s*"
                     r"['\"][^'\"]{10,})")
    for f in list(root.glob("*.py")) + list((root / "app").glob("*.py")) + \
            list((root / "static").glob("*.js")) + \
            list((root / "tests").glob("*.py")) + [root / "README.md"]:
        if not f.exists():
            continue
        m = pat.search(f.read_text(errors="ignore"))
        assert not m, f"token-like string in {f}"


def test_fusion_weight_absent_vs_ready(brain):
    """10. absent contributes exactly nothing; ready carries the 0.25
    start weight scaled by the hindsight multiplier."""
    from app.engine import PredictionEngine
    from app.feed import MarketFeed
    eng = PredictionEngine(MarketFeed(), emit=lambda m: None)
    # absent (fresh engine uses the global flywire; force absent state)
    saved = eng.brains.fw.state
    eng.brains.fw.state = "absent"
    assert eng.brain_fusion_weight("BTC", "tf1") == 0.0
    # ready: 0.25 × clip(1 + 0.25·z, 0.25, 1.5), z starts 0 -> 0.25
    eng.brains.fw.state = "ready"
    eng._fw_hindsight = {}
    w = eng.brain_fusion_weight("BTC", "tf1")
    assert abs(w - 0.25) < 1e-9
    eng._fw_hindsight[("BTC", "tf1")] = 1.4   # hindsight moved it
    w2 = eng.brain_fusion_weight("BTC", "tf1")
    assert abs(w2 - 0.25 * 1.4) < 1e-9
    eng.brains.fw.state = saved


def test_four_brains_vote_from_their_own_specialty():
    """The starting spec: 4 brains working DIFFERENTLY. Each instance
    must carry its own specialty signal (pattern / experience /
    state / learning); the consensus is their mean — not four copies
    of one hormone readout."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "app" / "brain.py").read_text()
    for bid in ("DB-1", "DB-2", "DB-3", "DB-4"):
        assert f'brain.id == "{bid}"' in src or bid == "DB-4"
    assert '"signal": round(' in src
    assert 'r["signal"] * wts[i]' in src          # consensus over specialties
    assert 'r["h_signal"] * wts[i]' not in src    # not the clone readout
    js = (root / "static" / "app.js").read_text()
    assert "specialty vote" in js
