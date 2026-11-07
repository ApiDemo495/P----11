"""FlyWire v783 whole-brain connectome — the brain component (brain.md).

The brain is the REAL connectome or it is absent: there is no substitute
matrix of any size (error B-3). This module downloads and caches the
FlyWire v783 edge table + neuron annotations, builds a normalised CSR
weight matrix once (in a worker process — B-1), verifies itself in five
steps, presents the market as an odour to the fly's own antennal-lobe
projection neurons, propagates through the real wiring, and reads the
fly's own approach/avoid balance as the vote — never overridden by our
sign (B-10), learned only through dopamine reward-prediction error.

Artefact adaptations (documented, not silent):
  * philshiu/Drosophila_brain_model's Connectivity_783.parquet ships
    `Presynaptic_ID / Postsynaptic_ID / Connectivity / Excitatory`
    (±1 sign) instead of the spec's `pre/post/syn_count/nt_type`; the
    upstream pipeline already folded neurotransmitter identity into the
    Excitatory flag, so sign := Excitatory, magnitude := Connectivity.
  * annotations super_class value is `descending` (not
    `descending_neuron`); top_nt uses full names (acetylcholine, gaba,
    glutamate, …).
  * approach = MBONs with top_nt in {acetylcholine, gaba}, avoid =
    glutamate (Aso et al. 2014 semantics preserved).
  * the two brain formulas (CCSv2, KCAE) are brain-channel outputs with
    the same trace/zero_reason conventions as F1..F31 but are NOT added
    to the core registry: the registry is contractually 31 formulas
    (weightage, tests, UI tables) — they surface on the brain card and
    in /api/brain/status instead.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

BUILD_VERSION = 1
CACHE_DIR = Path(
    os.environ.get("CONNECTOME_CACHE",
                   str(Path.home() / ".cache" / "drosophila-trader"
                       / "connectome")))
PARQUET = "Connectivity_783.parquet"
ANNOTATIONS = "Supplemental_file1_neuron_annotations.tsv"
EXPECTED_MIN = {PARQUET: 90_000_000, ANNOTATIONS: 25_000_000}
_SOURCES = {
    PARQUET: [
        "https://api.github.com/repos/philshiu/Drosophila_brain_model"
        "/contents/Connectivity_783.parquet",
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model"
        "/main/Connectivity_783.parquet",
    ],
    ANNOTATIONS: [
        "https://api.github.com/repos/flyconnectome/flywire_annotations"
        "/contents/supplemental_files/"
        "Supplemental_file1_neuron_annotations.tsv",
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations"
        "/main/supplemental_files/"
        "Supplemental_file1_neuron_annotations.tsv",
    ],
}
RAW_ACCEPT = {"Accept": "application/vnd.github.raw"}

# MB sub-circuit (brain.md §2.5) — the fast path, milliseconds per pass
MB_CLASSES = ("ALPN", "Kenyon_Cell", "DAN", "MBON", "MBIN", "LHLN",
              "LHCENT")
APPROACH_NT = ("acetylcholine", "gaba")
AVOID_NT = ("glutamate",)
DN_SUPER = "descending"
KC_SPARSITY = 0.05            # APL feedback: keep top 5 % of Kenyon cells
VALENCE_HOPS = 3
# N_INPUTS: this app's signed directional formula outputs (F1..F31 minus
# F20 DRHE, the confidence gauge) — one odour channel per formula.
N_INPUTS = 30
WHOLE_PASS_PERIOD_S = 2.0     # whole-brain pass on a worker thread (B-6)
RPE_LAMBDA = 2.25
RPE_TONIC = 0.85
RPE_PHASIC_DECAY = 0.6


def _log(msg: str) -> None:
    print(f"flywire: {msg}", flush=True)


# --------------------------------------------------------------------------
# acquisition (resume + size check; short file = failed download, B-7)
# --------------------------------------------------------------------------
def acquire() -> dict[str, str]:
    """Fetch the two artefacts into CACHE_DIR. Returns {name: state}."""
    import httpx
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name, urls in _SOURCES.items():
        dest = CACHE_DIR / name
        want = EXPECTED_MIN[name]
        if dest.exists() and dest.stat().st_size >= want:
            out[name] = "cached"
            continue
        got = False
        for url in urls:
            tmp = dest.with_suffix(dest.suffix + ".part")
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = dict(RAW_ACCEPT) if "api.github.com" in url else {}
            if have and have < want:
                headers["Range"] = f"bytes={have}-"
            try:
                with httpx.stream("GET", url, headers=headers, timeout=60.0,
                                  follow_redirects=True) as r:
                    if r.status_code in (200, 206):
                        mode = "ab" if (r.status_code == 206 and have) else "wb"
                        if mode == "wb":
                            have = 0
                        with open(tmp, mode) as f:
                            for chunk in r.iter_bytes(1 << 20):
                                f.write(chunk)
                        if tmp.stat().st_size >= want:
                            tmp.rename(dest)
                            out[name] = "downloaded"
                            got = True
                            break
            except Exception as exc:              # try next source
                _log(f"{name}: {url[:60]}… failed ({exc})")
            # curl fallback (B-7): some sandboxes intercept TLS in a way
            # certifi cannot see while curl's system CA bundle can —
            # the size check below keeps a short/corrupt file a FAILURE
            if not got:
                try:
                    import subprocess
                    cmd = ["curl", "-sL", "-m", "600", "-o", str(tmp)]
                    for hk, hv in headers.items():
                        cmd += ["-H", f"{hk}: {hv}"]
                    cmd.append(url)
                    rc = subprocess.call(
                        cmd,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if rc == 0 and tmp.exists() and tmp.stat().st_size >= want:
                        tmp.rename(dest)
                        out[name] = "downloaded (curl)"
                        got = True
                except Exception as exc:
                    _log(f"{name}: curl fallback failed ({exc})")
        if not got:
            size = dest.stat().st_size if dest.exists() else 0
            out[name] = (f"failed (have {size} of ≥{want} bytes)")
    return out


# --------------------------------------------------------------------------
# build (worker process — B-1: a crash never takes the API down)
# --------------------------------------------------------------------------
def _build_worker(cache_dir: str, out_npz: str, pipe) -> None:
    """Spawn target (B-1): progress over the pipe, errors reported, never
    crashes the API process."""
    try:
        def progress(stage: str, pct: float) -> None:
            try:
                pipe.send({"stage": stage, "pct": round(pct, 1)})
            except Exception:
                pass
        build_sync(Path(cache_dir) / PARQUET,
                   Path(cache_dir) / ANNOTATIONS,
                   Path(out_npz), progress=progress)
        pipe.send({"done": True})
    except Exception as exc:                        # pragma: no cover
        try:
            pipe.send({"error": repr(exc)})
        except Exception:
            pass


def build_sync(parquet_path: Path, ann_path: Path, out_npz: Path,
               progress=None) -> dict:
    """Synchronous build against explicit paths (tests + worker reuse).

    Shares the worker's column-at-a-time strategy; kept separate so the
    acceptance tests can build from a 200-row synthetic parquet without
    spawning processes.
    """
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    from scipy import sparse

    def prog(stage: str, pct: float) -> None:
        if progress:
            progress(stage, pct)

    prog("reading edges", 5)
    pre = pq.read_table(parquet_path,
                        columns=["Presynaptic_ID"]).column(0).to_numpy()
    post = pq.read_table(parquet_path,
                         columns=["Postsynaptic_ID"]).column(0).to_numpy()
    conn = pq.read_table(parquet_path,
                         columns=["Connectivity"]).column(0).to_numpy()
    conn = np.clip(conn, 0, 65535).astype(np.uint16)
    sign = pq.read_table(parquet_path,
                         columns=["Excitatory"]).column(0).to_numpy()
    sign = np.where(sign >= 0, 1, -1).astype(np.int8)
    prog("mapping ids", 35)
    roots, inv = np.unique(np.concatenate([pre, post]),
                           return_inverse=True)
    n = len(roots)
    src = inv[:len(pre)].astype(np.int64)
    dst = inv[len(pre):].astype(np.int64)
    del pre, post, inv
    prog("normalising", 50)
    total_in = np.bincount(dst, weights=conn.astype(np.float64),
                           minlength=n)
    total_in[total_in == 0] = 1.0
    data = (sign.astype(np.float32)
            * (conn.astype(np.float32)
               / total_in[dst].astype(np.float32)))
    W = sparse.csr_matrix((data, (dst, src)), shape=(n, n))
    W.sort_indices()
    del data, src, dst, conn, sign, total_in
    prog("annotations", 65)
    ann = pcsv.read_csv(
        ann_path,
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(
            include_columns=["root_id", "cell_class", "super_class",
                             "top_nt", "cell_type"]))
    a_root = ann.column("root_id").to_numpy().astype(np.int64)
    a_cls = np.array([str(v) for v in ann.column("cell_class")], dtype=object)
    a_sup = np.array([str(v) for v in ann.column("super_class")], dtype=object)
    a_nt = np.array([str(v) for v in ann.column("top_nt")], dtype=object)
    a_type = np.array([str(v) for v in ann.column("cell_type")], dtype=object)
    order = np.argsort(a_root, kind="stable")
    a_root = a_root[order]
    a_cls, a_sup, a_nt, a_type = (a_cls[order], a_sup[order], a_nt[order],
                                  a_type[order])
    pos = np.searchsorted(a_root, roots)
    pos[pos >= len(a_root)] = 0
    matched = a_root[pos] == roots
    cls = np.where(matched, a_cls[pos], "").astype(object)
    sup = np.where(matched, a_sup[pos], "").astype(object)
    nt = np.where(matched, a_nt[pos], "").astype(object)
    ctype = np.where(matched, a_type[pos], "").astype(object)
    del ann, a_root, a_cls, a_sup, a_nt, a_type, order, pos, matched
    prog("sub-circuit", 80)
    mb_mask = np.isin(cls, np.array(MB_CLASSES, dtype=object))
    mb_idx = np.flatnonzero(mb_mask)
    Wm = W[mb_idx][:, mb_idx]
    prog("groups", 88)
    alpn_glob = mb_idx[cls[mb_idx] == "ALPN"]
    akey = np.array([f"{ctype[i]}|{roots[i]}" for i in alpn_glob],
                    dtype=object)
    alpn_glob = alpn_glob[np.argsort(akey, kind="stable")]
    mbon_glob = mb_idx[cls[mb_idx] == "MBON"]
    mbon_nt = nt[mbon_glob]
    dan_glob = mb_idx[cls[mb_idx] == "DAN"]
    kc_glob = mb_idx[cls[mb_idx] == "Kenyon_Cell"]
    dn_mask = sup == DN_SUPER
    dan_type = np.array([str(ctype[i]).split("_")[0].split("-")[0]
                         for i in dan_glob], dtype=object)
    oa_glob = mb_idx[nt[mb_idx] == "octopamine"]
    lh_glob = mb_idx[np.isin(cls[mb_idx],
                            np.array(["LHLN", "LHCENT"], dtype=object))]
    prog("saving", 95)
    np.savez(
        out_npz,
        build_version=np.int64(BUILD_VERSION),
        roots=roots.astype(np.int64),
        W_data=W.data, W_indices=W.indices.astype(np.int64),
        W_indptr=W.indptr.astype(np.int64),
        W_shape=np.array(W.shape, dtype=np.int64),
        mb_idx=mb_idx.astype(np.int64),
        mb_data=Wm.data, mb_indices=Wm.indices.astype(np.int64),
        mb_indptr=Wm.indptr.astype(np.int64),
        mb_shape=np.array(Wm.shape, dtype=np.int64),
        alpn_glob=alpn_glob.astype(np.int64),
        alpn_type=np.array([str(ctype[i]) for i in alpn_glob]),
        mbon_glob=mbon_glob.astype(np.int64),
        mbon_nt=np.array([str(v) for v in mbon_nt]),
        dan_glob=dan_glob.astype(np.int64),
        dan_type=np.array([str(v) for v in dan_type]),
        oa_glob=oa_glob.astype(np.int64),
        lh_glob=lh_glob.astype(np.int64),
        kc_glob=kc_glob.astype(np.int64),
        dn_mask=dn_mask.astype(np.bool_),
        cls=np.array([str(v) for v in cls]),
        meta=np.array(json.dumps({
            "build_version": BUILD_VERSION,
            "neurons": int(n), "edges": int(W.nnz),
            "mb_neurons": int(len(mb_idx)), "mb_edges": int(Wm.nnz),
            "alpn": int(len(alpn_glob)), "mbon": int(len(mbon_glob)),
            "kc": int(len(kc_glob)), "dan": int(len(dan_glob)),
            "dn": int(dn_mask.sum()), "built_at": time.time(),
        })),
    )
    prog("cached", 100)
    return json.loads(str(np.load(out_npz, allow_pickle=False)["meta"]))


def start_build() -> mp.Process:
    """Spawn the build worker (B-1/B-11). Returns the Process.

    SPAWN, not fork: forking the whole API server (feed buffers, rings,
    any half-built scipy state) into the build child is fragile and
    bloats the child's resident set — on a fresh 4-core Codespace the
    forked worker died silently (no pipe message, no cache file, just
    "build produced no cache file"). A spawned child re-imports clean
    and holds only the build's own arrays."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parent, child = mp.Pipe(duplex=False)
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_build_worker,
                    args=(str(CACHE_DIR), str(CACHE_DIR / "flywire_783.npz"),
                          child),
                    daemon=True)
    p.start()
    start_build.last_pipe = parent          # type: ignore[attr-defined]
    return p


# --------------------------------------------------------------------------
# the brain itself
# --------------------------------------------------------------------------
_GLOBAL: FlyWireBrain | None = None


def get_flywire() -> "FlyWireBrain":
    """One connectome per process (load happens once)."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = FlyWireBrain()
    return _GLOBAL



def short_error(e) -> str | None:
    """One-line human summary of a brain failure. Raw exception text
    (thrift deserialize / page-header fragments) read as 'millions of
    errors' in the UI — it is never surfaced again."""
    if not e:
        return None
    t = str(e).lower()
    if "thrift" in t or "deserialize" in t or "page header" in t or "pageheader" in t:
        return "neuPrint dataset unreadable here (cached file format) — RECONNECT rebuilds it"
    if "oom" in t or "exit -" in t or "out of memory" in t:
        return "build ran out of memory (~2 GB free RAM needed) — RECONNECT retries"
    if "download" in t or "network" in t or "timed out" in t or "timeout" in t:
        return "download interrupted — RECONNECT retries"
    if "token" in t or "401" in t or "403" in t or "auth" in t:
        return "neuPrint token missing/rejected — paste one in Settings → Brain"
    return "brain offline — vote withheld; Settings → Brain → RECONNECT"

class FlyWireBrain:
    """Download → build → verify → encode → propagate → vote → learn.

    State machine: absent → building → ready (hot-swap, B-11). Nothing
    here ever fabricates a matrix: if the connectome is not loaded the
    vote is withheld and the fusion weight is 0 (B-3).
    """

    def __init__(self) -> None:
        self.state = "absent"               # absent | building | ready
        self.progress: dict[str, Any] = {"stage": "not started", "pct": 0.0}
        self.last_error: str | None = None
        self.steps: list[dict] = []
        self.load_ms: float | None = None
        self.pass_age_s: float | None = None
        self.last_pass_ms: float | None = None
        self.last_vote: float | None = None
        self.last_conf: float | None = None
        self.last_dn_share: float | None = None
        self.dead_channels: list[int] = []
        self.flips: np.ndarray | None = None       # +1 / -1 per channel
        # dopamine RPE state (§7)
        self.E = 0.0                                # expectation (EMA)
        self.drg = 0.0                              # dopamine release gain
        self.phasic = 0.0
        self.tonic = RPE_TONIC
        self.alpha = 0.10
        self.rpe_history: list[dict] = []
        self._kc_mbon_delta: np.ndarray | None = None
        self._rollback_snapshot: dict | None = None
        self._kcae_ring: list[float] = []
        self._last_values: list[float] | None = None
        self._anatomy: dict | None = None
        self._anatomy_state = "idle"      # idle | computing | ready | error
        self._last_mb: dict | None = None
        self._last_whole: dict | None = None
        self._anatomy_error = ""
        self._last_active_kc: np.ndarray | None = None
        self._base_kc_mbon: np.ndarray | None = None
        self._proc: mp.Process | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        # matrices (None until ready)
        self.W = None                             # scipy csr, whole brain
        self.Wm = None                            # MB sub-circuit csr
        self.meta: dict = {}
        self.roots: np.ndarray | None = None
        self.mb_idx: np.ndarray | None = None
        self.alpn: np.ndarray | None = None       # global ids, sorted
        self.alpn_type: np.ndarray | None = None
        self.mbon: np.ndarray | None = None       # global ids
        self.mbon_nt: np.ndarray | None = None
        self.dan: np.ndarray | None = None
        self.kc: np.ndarray | None = None
        self.dn_mask: np.ndarray | None = None
        self.cls: np.ndarray | None = None
        # local (MB) index maps
        self._l_of: np.ndarray | None = None      # global -> mb-local
        self._alpn_local: np.ndarray | None = None
        self._mbon_local: np.ndarray | None = None
        self._kc_local: np.ndarray | None = None
        self._dan_local: np.ndarray | None = None
        self._mbon_approach: np.ndarray | None = None
        self._mbon_avoid: np.ndarray | None = None

    # ------------------------------------------------------------- lifecycle
    def npz_path(self) -> Path:
        return CACHE_DIR / "flywire_783.npz"

    def _purge_corrupt(self, exc) -> bool:
        """A truncated/corrupt download used to poison EVERY rebuild:
        the same unreadable parquet was retried forever ("brain absent
        again and again for a day"). If the failure smells like a bad
        artefact (thrift/deserialize/parquet), DELETE the cached files
        so the next attempt re-downloads clean ones."""
        t = str(exc).lower()
        if not any(k in t for k in ("thrift", "deserialize", "parquet",
                                    "page header", "magic", "stream")):
            return False
        purged = []
        for name in (PARQUET, ANNOTATIONS):
            for cand in (CACHE_DIR / name,
                         (CACHE_DIR / name).with_suffix(
                             (CACHE_DIR / name).suffix + ".part")):
                try:
                    if cand.exists():
                        cand.unlink(); purged.append(str(cand))
                except OSError:
                    pass
        try:
            if self.npz_path().exists():
                self.npz_path().unlink(); purged.append("npz")
        except OSError:
            pass
        if purged:
            _log(f"purged corrupt cache artefacts: {purged}")
        return bool(purged)

    def _schedule_retry(self) -> None:
        """Bounded auto-retry when a build fails: up to 5 attempts,
        3 minutes apart — a flaky download self-heals without the user
        hunting for a RECONNECT button. Manual RECONNECT resets it."""
        self._retry_n = getattr(self, "_retry_n", 0) + 1
        if self._retry_n > 5:
            _log("brain auto-retry exhausted (5) — use RECONNECT")
            return
        def _later():
            time.sleep(180)
            if self.state == "absent":
                _log(f"brain auto-retry {self._retry_n}/5")
                self.ensure()
        threading.Thread(target=_later, name="flywire-retry",
                         daemon=True).start()

    def ensure(self) -> None:
        """Hot-swap in the connectome without ever blocking first paint."""
        with self._lock:
            if self.state in ("building", "ready"):
                return
            npz = self.npz_path()
            if npz.exists():
                try:
                    t0 = time.perf_counter()
                    self._load(npz)
                    self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
                    self.state = "ready"
                    self.last_error = None
                    self._start_pass_thread()
                    self.verify()
                    _log(f"cached build loaded in {self.load_ms} ms "
                         f"({self.meta.get('neurons')} neurons)")
                    return
                except Exception as exc:
                    self.last_error = f"load failed: {exc!r}"
                    _log(self.last_error)
                    self._purge_corrupt(exc)
            # no valid cache: acquire + build in a worker (B-11)
            self.state = "building"
            self.progress = {"stage": "starting", "pct": 0.0}
            threading.Thread(target=self._acquire_then_build,
                             name="flywire-acquire", daemon=True).start()

    def _acquire_then_build(self) -> None:
        try:
            # download with REAL progress: poll the .part files while
            # acquire() runs (the old code sat at "1% — downloading"
            # for the whole 130 MB, indistinguishable from a hang)
            self.progress = {"stage": "downloading", "pct": 1.0}
            _res: dict = {}
            _th = threading.Thread(
                target=lambda: _res.update(states=acquire()),
                name="flywire-download", daemon=True)
            _th.start()
            _want = float(sum(EXPECTED_MIN.values())) or 1.0
            while _th.is_alive():
                _have = 0.0
                for _n in _SOURCES:
                    for _cand in (CACHE_DIR / _n,
                                  (CACHE_DIR / _n).with_suffix(
                                      (CACHE_DIR / _n).suffix + ".part")):
                        try:
                            if _cand.exists():
                                _have += _cand.stat().st_size
                        except OSError:
                            pass
                self.progress = {"stage": "downloading",
                                 "pct": round(min(40.0, 1.0 + 39.0
                                                  * _have / _want), 1)}
                _th.join(timeout=0.5)
            states = _res.get("states") or {}
            # curl fallback returns "downloaded (curl)" — anything that
            # actually landed counts as success (was marking a FULL
            # download as failed → brain stayed absent)
            bad = {k: v for k, v in states.items()
                   if not (v == "cached" or str(v).startswith("downloaded"))}
            if bad:
                self.state = "absent"
                self.last_error = ("download failed: "
                                   + "; ".join(f"{k} {v}"
                                               for k, v in bad.items()))
                _log(self.last_error)
                self.verify()
                return
            self.progress = {"stage": "building", "pct": 2.0}
            self._proc = start_build()
            pipe = getattr(start_build, "last_pipe", None)
            while self._proc.is_alive():
                if pipe is not None and pipe.poll(0.5):
                    msg = pipe.recv()
                    if msg.get("error"):
                        raise RuntimeError(msg["error"])
                    self.progress = {"stage": msg.get("stage", "building"),
                                     "pct": float(msg.get("pct", 0.0))}
                    if msg.get("done"):
                        break
                else:
                    time.sleep(0.4)
            self._proc.join(timeout=5)
            if not self.npz_path().exists():
                _rc = self._proc.exitcode
                raise RuntimeError(
                    f"build worker died (exit {_rc}) without writing "
                    "the cache file — a negative exit is a signal "
                    "(-9 = OOM-killed: the build needs ~2 GB free RAM; "
                    "close other Codespace processes) — press RECONNECT "
                    "to retry; the vote stays withheld meanwhile")
            t0 = time.perf_counter()
            self._load(self.npz_path())
            self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
            self.state = "ready"
            self.last_error = None
            self.progress = {"stage": "cached", "pct": 100.0}
            self._start_pass_thread()
            self.verify()
            _log(f"build complete, hot-swapped in {self.load_ms} ms")
        except Exception as exc:
            self.state = "absent"
            self.last_error = repr(exc)
            self.progress = {"stage": "failed", "pct": 0.0}
            _log(f"build failed: {exc!r}")
            self._purge_corrupt(exc)
            self._schedule_retry()
            self.verify()

    def rebuild(self) -> dict:
        """POST /api/brain/reconnect — drop cache state and start over."""
        with self._lock:
            if self.state == "building":
                return {"ok": False, "state": "building",
                        "progress": self.progress}
            self.state = "absent"
            self.last_error = None
            self._retry_n = 0
        self.ensure()
        return {"ok": True, "state": self.state}

    # ------------------------------------------------------------- loading
    def _load(self, npz: Path) -> None:
        from scipy import sparse
        z = np.load(npz, allow_pickle=False)
        if int(z["build_version"]) != BUILD_VERSION:
            raise ValueError("BUILD_VERSION mismatch (B-8) — rebuild")
        self.W = sparse.csr_matrix(
            (z["W_data"], z["W_indices"], z["W_indptr"]),
            shape=tuple(z["W_shape"]))
        self.Wm = sparse.csr_matrix(
            (z["mb_data"], z["mb_indices"], z["mb_indptr"]),
            shape=tuple(z["mb_shape"]))
        self.meta = json.loads(str(z["meta"]))
        self.roots = z["roots"]
        self.mb_idx = z["mb_idx"]
        self.alpn = z["alpn_glob"]
        self.alpn_type = z["alpn_type"]
        self.mbon = z["mbon_glob"]
        self.mbon_nt = z["mbon_nt"]
        self.dan = z["dan_glob"]
        self.dan_type = z["dan_type"]
        self.oa = z["oa_glob"]
        self.lh = z["lh_glob"]
        self.kc = z["kc_glob"]
        self.dn_mask = z["dn_mask"]
        self.cls = z["cls"]
        self._l_of = np.full(len(self.roots), -1, dtype=np.int64)
        self._l_of[self.mb_idx] = np.arange(len(self.mb_idx))
        self._alpn_local = self._l_of[self.alpn]
        self._mbon_local = self._l_of[self.mbon]
        self._kc_local = self._l_of[self.kc]
        self._dan_local = self._l_of[self.dan]
        self._oa_local = self._l_of[self.oa]
        self._lh_l = self._l_of[self.lh]
        self._mbon_approach = np.isin(self.mbon_nt,
                                      np.array(APPROACH_NT))
        self._mbon_avoid = np.isin(self.mbon_nt, np.array(AVOID_NT))
        if self.flips is None or len(self.flips) != self.n_channels():
            self.flips = np.ones(self.n_channels(), dtype=np.int8)
        if self._kc_mbon_delta is None or \
                self._kc_mbon_delta.shape != (len(self.kc),
                                              len(self.mbon)):
            self._kc_mbon_delta = np.zeros((len(self.kc), len(self.mbon)),
                                           dtype=np.float32)
        # polarity calibration is a property of the wiring (B-2)
        if not self.dead_channels and not getattr(self, "_calibrated", False):
            self.calibrate()
        # base KC->MBON slice for RPE updates
        self._base_kc_mbon = self._extract_kc_mbon()
        # warm the pyarrow import HERE, on the main thread: the anatomy
        # thread also imports it, and a first-import race between two
        # threads surfaces as "partially initialized module pyarrow"
        try:
            import pyarrow.csv    # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except Exception:
            pass
        # apply a vault restore that arrived before the matrices existed
        pend = getattr(self, "_pending_state", None)
        if pend:
            self._pending_state = None
            self.load_state(pend)

    def n_channels(self) -> int:
        return min(N_INPUTS, max(1, len(self.alpn) // 2))

    def _extract_kc_mbon(self) -> np.ndarray:
        """Dense (n_kc × n_mbon) base weights from the MB CSR."""
        assert self.Wm is not None
        kc_l = self._kc_local
        mb_l = self._mbon_local
        sub = self.Wm[kc_l][:, mb_l]              # rows KC, cols MBON
        return np.asarray(sub.todense(), dtype=np.float32)

    # ------------------------------------------------------ verification
    def verify(self) -> list[dict]:
        """Five steps, every outcome logged and shown (brain.md §3)."""
        now = time.time()
        npz = self.npz_path()
        steps: list[dict] = []

        def step(name: str, ok: bool, detail: str,
                 ms: float | None = None) -> None:
            steps.append({"step": name, "ok": ok, "detail": detail,
                          "ms": ms})

        t0 = time.perf_counter()
        cached = (npz.exists()
                  and npz.stat().st_size > 0
                  and self.state == "ready")
        step("cache", cached,
             (f"cached (v{BUILD_VERSION}) · "
              f"{npz.stat().st_size // (1 << 20)} MB" if npz.exists()
              else "no npz"),
             round((time.perf_counter() - t0) * 1000, 1))
        # steps 1-3: neuPrint reach/auth/query — OPTIONAL (FlyWire is the
        # primary source); only attempted when a key exists (memory only)
        neuprint = self._neuprint_steps()
        steps.extend(neuprint)
        t0 = time.perf_counter()
        fly_ok = self.state == "ready"
        detail = (f"FlyWire v783 ready "
                  f"({self.meta.get('neurons', 0):,} neurons · "
                  f"{self.meta.get('edges', 0):,} connections · "
                  f"MB {self.meta.get('mb_neurons', 0):,})"
                  if fly_ok else
                  (self.last_error
                   or (f"building {self.progress.get('pct', 0):.0f} %"
                       if self.state == "building"
                       else "not loaded")))
        step("flywire", fly_ok, detail,
             self.load_ms or round((time.perf_counter() - t0) * 1000, 1))
        absent = not fly_ok
        step("absent", True if not absent else True,
             ("BRAIN ABSENT — vote withheld, retrying in 60 s"
              if absent else "connectome loaded — vote live"))
        self.steps = steps
        return steps

    def _neuprint_steps(self) -> list[dict]:
        """Optional neuPrint reach/auth/query (memory-only key, B-5)."""
        key = os.environ.get("NEUPRINT_TOKEN") or ""
        if not key:
            return [{"step": "neuprint reach", "ok": True,
                     "detail": "no key (optional — FlyWire is primary)"},
                    {"step": "neuprint auth", "ok": True,
                     "detail": "no key"},
                    {"step": "neuprint query", "ok": True,
                     "detail": "skipped (no key)"}]
        import httpx
        out = []
        t0 = time.time()
        try:
            r = httpx.get("https://neuprint.janelia.org/api/dbmeta/datasets",
                          headers={"Authorization": f"Bearer {key}"},
                          timeout=8.0)
            out.append({"step": "neuprint reach", "ok": True,
                        "detail": "neuprint reachable",
                        "ms": round((time.time() - t0) * 1000, 1)})
            if r.status_code == 401:
                out.append({"step": "neuprint auth", "ok": False,
                            "detail": "invalid key"})
                out.append({"step": "neuprint query", "ok": False,
                            "detail": "skipped (invalid key)"})
            else:
                out.append({"step": "neuprint auth", "ok": r.ok,
                            "detail": "token accepted" if r.ok
                            else f"HTTP {r.status_code}"})
                out.append({"step": "neuprint query", "ok": r.ok,
                            "detail": "datasets listed" if r.ok
                            else "query failed"})
        except httpx.TimeoutException:
            out.append({"step": "neuprint reach", "ok": False,
                        "detail": "unreachable (timeout 8 s)"})
            out.append({"step": "neuprint auth", "ok": False,
                        "detail": "skipped (unreachable)"})
            out.append({"step": "neuprint query", "ok": False,
                        "detail": "skipped (unreachable)"})
        except Exception as exc:
            out.append({"step": "neuprint reach", "ok": False,
                        "detail": f"error: {exc}"})
            out.append({"step": "neuprint auth", "ok": False,
                        "detail": "skipped"})
            out.append({"step": "neuprint query", "ok": False,
                        "detail": "skipped"})
        return out

    def chip(self) -> dict:
        """UI brain chip: green ready / blue building / red absent."""
        if self.state == "ready":
            age = (round(time.time() - self.last_pass_at)
                   if getattr(self, "last_pass_at", 0) else None)
            return {"kind": "ok",
                    "text": (f"FlyWire v783 · "
                             f"{self.meta.get('neurons', 0):,} n · "
                             f"pass {self.last_pass_ms or 0:.0f} ms"
                             + (f" · {age} s ago" if age is not None
                                else ""))}
        if self.state == "building":
            _live = (getattr(self, "last_pass_at", 0)
                     and time.time() - self.last_pass_at < 30)
            return {"kind": "build",
                    "text": (f"building {self.progress.get('pct', 0):.0f} %"
                             f" — {self.progress.get('stage', '')}"
                             + (" · vote live from current build"
                                if _live else ""))}
        return {"kind": "absent",
                "text": "brain offline — vote withheld "
                        "(Settings → Brain → RECONNECT)"}

    # ---------------------------------------------------------- encoding
    def encode(self, values: list[float]) -> np.ndarray | None:
        """Push–pull glomerular rates: ON = max(0, x), OFF = max(0, -x).

        Polarity flips (property of the wiring) are applied here so the
        circuit's innate aversion per channel never silently inverts a
        formula's meaning (B-2).
        """
        if self.state != "ready" or self.alpn is None:
            return None
        n = self.n_channels()
        x = np.zeros(n, dtype=np.float32)
        for i in range(n):
            v = float(values[i]) if i < len(values) else 0.0
            v = max(-1.0, min(1.0, v))
            if self.flips is not None:
                v *= float(self.flips[i])
            x[i] = v
        rates = np.zeros(2 * n, dtype=np.float32)
        rates[:n] = np.maximum(x, 0.0)
        rates[n:] = np.maximum(-x, 0.0)
        return rates

    def _inject(self, rates: np.ndarray) -> np.ndarray:
        """Spread 2·N_INPUTS glomerular groups over the sorted ALPNs."""
        h0 = np.zeros(len(self.mb_idx), dtype=np.float32)
        alpn_l = self._alpn_local
        n_groups = len(rates)
        # contiguous deterministic groups over the sorted ALPN list
        bounds = np.linspace(0, len(alpn_l), n_groups + 1).astype(int)
        for g in range(n_groups):
            seg = alpn_l[bounds[g]:bounds[g + 1]]
            if len(seg):
                h0[seg] = rates[g]
        return h0

    # ------------------------------------------------------- propagation
    def pass_mb(self, rates: np.ndarray | None,
                drg: float | None = None, hsi: float = 0.0) -> dict:
        """MB sub-circuit pass: 3 hops, KC top-5 % (APL), dopamine /
        octopamine gains. Milliseconds — safe per evaluation (B-6)."""
        if self.state != "ready" or rates is None:
            return {"ok": False, "reason": "brain absent"}
        from scipy import sparse  # noqa: F401  (csr matvec)
        t0 = time.perf_counter()
        drg = self.drg if drg is None else drg
        Wm = self.Wm
        h = self._inject(rates)
        kc_l = self._kc_local
        dan_l = self._dan_local
        mbon_acc = np.zeros(len(self.mbon), dtype=np.float32)
        lh_acc = 0.0
        kc_act = None
        for hop in range(VALENCE_HOPS):
            h_prev = h
            h = Wm @ h
            # learned KC->MBON overlay (dopamine RPE, §7): the base
            # matrix is never retrained globally — only this delta moves
            if self._kc_mbon_delta is not None and np.any(h_prev[kc_l]):
                h[self._mbon_local] += (
                    self._kc_mbon_delta.T @ h_prev[kc_l])
            np.maximum(h, 0.0, out=h)               # ReLU
            # APL feedback: keep top 5 % of Kenyon cells EVERY hop (B-9)
            if len(kc_l):
                k = max(1, int(math.ceil(KC_SPARSITY * len(kc_l))))
                vals = h[kc_l]
                thresh = np.partition(vals, -k)[-k]
                keep = np.zeros_like(vals)
                mask = vals >= thresh
                keep[mask] = vals[mask]
                h[kc_l] = keep
                if hop == 0:
                    kc_act = keep.copy()
                    # active set for the boundary RPE update (§7)
                    self._last_active_kc = np.flatnonzero(keep > 0)
            # dopamine gains: PAM (reward) up with +drg, PPL1 up with -drg
            if dan_l is not None and len(dan_l):
                dan_types = np.array([str(t) for t in self._dan_types()])
                # cell_type prefixes come as PAM01…/PPL101… (build-time
                # split keeps the digits), so match on the prefix
                pam = np.char.startswith(dan_types, "PAM")
                ppl1 = np.char.startswith(dan_types, "PPL1")
                gains = np.ones(len(dan_l), dtype=np.float32)
                gains[pam] += 0.5 * max(0.0, float(drg))
                gains[ppl1] += 0.5 * max(0.0, -float(drg))
                h[dan_l] *= gains
            # octopamine ← HSI / stress index
            if self._oa_local is not None and len(self._oa_local):
                h[self._oa_local] *= (1.0 + 0.3 * float(hsi))
            mbon_acc += h[self._mbon_local]
        # lateral-horn + MBON readout (CCSv2)
        lh_l = self._lh_l
        lh_act = float(h[lh_l].sum()) if len(lh_l) else 0.0
        approach = float(mbon_acc[self._mbon_approach].sum())
        avoid = float(mbon_acc[self._mbon_avoid].sum())
        denom = abs(approach) + abs(avoid) + 1e-9
        balance = math.tanh(2.0 * (approach - avoid) / denom)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        kc_active = int(np.count_nonzero(h[kc_l])) if kc_l is not None else 0
        self._last_mb = {
            "circuit": "FlyWire v783 mushroom body + lateral horn",
            "approach": round(approach, 6), "avoid": round(avoid, 6),
            "balance": round(balance, 6),
            "lh": round(lh_act, 6),
            "kc_active": int(np.count_nonzero(kc_act)) if kc_act is not None else 0,
            "kc_total": int(len(self.kc)),
            "ms": ms,
        }
        # hop-1 sparsity is what APL enforces; report both
        hop1_active = int(np.count_nonzero(kc_act)) if kc_act is not None else 0
        return {"ok": True, "ms": ms, "balance": balance,
                "approach": approach, "avoid": avoid, "lh": lh_act,
                "mbon_acc": mbon_acc, "kc_act": kc_act,
                "kc_active_hop1": hop1_active, "kc_active_final": kc_active}

    def _dan_types(self) -> np.ndarray:
        """PAM / PPL1 prefixes saved at build time (cell_type)."""
        return getattr(self, "dan_type", np.array([], dtype=object))


    # ------------------------------------------------------------ readout
    def _balance(self, rates: np.ndarray, **kw) -> float:
        r = self.pass_mb(rates, **kw)
        return float(r.get("balance", 0.0)) if r.get("ok") else 0.0

    def vote(self, values: list[float], **kw) -> dict:
        """Push–pull mirrored vote: ½(f(x) − f(−x)) — antisymmetric by
        construction, exactly 0 on a flat market (honest gauge)."""
        if self.state != "ready":
            return {"ok": False, "reason": "brain absent", "vote": None}
        self._last_values = [float(v) for v in values]
        rates = self.encode(values)
        if rates is None or not np.any(rates):
            return {"ok": True, "vote": 0.0, "conf": None,
                    "reason": "flat input", "kind": "gauge"}
        r_pos = self.pass_mb(rates, **kw)
        bal_p = float(r_pos.get("balance", 0.0)) if r_pos.get("ok") else 0.0
        self.note_kc(r_pos.get("kc_act"))
        # f(−x): re-encode the negated INPUT (push-pull ON/OFF swap) —
        # negating the non-negative rate vector would zero the circuit
        rates_neg = self.encode([-v for v in values])
        bal_n = self._balance(rates_neg, **kw) if rates_neg is not None else 0.0
        v = 0.5 * (bal_p - bal_n)
        conf = self.last_conf                     # whole-brain DN share
        age = (round(time.time() - self.last_pass_at, 1)
               if getattr(self, "last_pass_at", 0) else None)
        return {"ok": True, "vote": round(float(v), 6),
                "conf": conf, "conf_age_s": age,
                "balance": round(bal_p, 6), "kind": "signed"}

    # ----------------------------------------------- polarity calibration
    def calibrate(self) -> dict:
        """Probe each channel alone at +1; record flips s_i = −1 where
        the circuit's mirrored balance is negative (innate aversion)."""
        if self.state != "ready":
            return {"ok": False, "reason": "brain absent"}
        n = self.n_channels()
        flips = np.ones(n, dtype=np.int8)
        probes: list[float] = []
        dead: list[int] = []
        for i in range(n):
            values = [0.0] * n
            values[i] = 1.0
            rates = self.encode_with_flips(values, flip_override=np.ones(n,
                                                                         dtype=np.int8))
            if rates is None or not np.any(rates):
                probes.append(0.0)
                dead.append(i)
                continue
            rates_neg = self.encode_with_flips(
                [-v for v in values],
                flip_override=np.ones(n, dtype=np.int8))
            bal = 0.5 * (self._balance(rates)
                         - self._balance(rates_neg))
            probes.append(round(bal, 6))
            if abs(bal) < 1e-12:
                dead.append(i)
            elif bal < 0:
                flips[i] = -1
        with self._lock:
            self.flips = flips
            self.dead_channels = dead
            self._calibrated = True
        return {"ok": True, "flips": flips.tolist(),
                "probes": probes, "dead": dead,
                "flipped": [int(i) for i in np.flatnonzero(flips < 0)]}

    def encode_with_flips(self, values, flip_override) -> np.ndarray | None:
        save = self.flips
        self.flips = flip_override
        try:
            return self.encode(values)
        finally:
            self.flips = save

    # -------------------------------------------------- whole-brain pass
    def _start_pass_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:                       # pragma: no cover - live
            while not self._stop.is_set():
                try:
                    if self.state == "ready" and self.W is not None:
                        t0 = time.perf_counter()
                        # whole-brain: same push-pull input over ALL ALPNs
                        n = self.n_channels()
                        values = list(self._last_values or [0.0] * n)
                        rates = self.encode(values)
                        if rates is not None:
                            h = self._inject_whole(rates)
                            hop_counts = []
                            for _ in range(VALENCE_HOPS):
                                h = self.W @ h
                                np.maximum(h, 0.0, out=h)
                                hop_counts.append(int(np.count_nonzero(h)))
                            dn = float(h[self.dn_mask].sum())
                            tot = float(h.sum()) + 1e-9
                            share = dn / tot
                            # full read-out (brain.md §12): approach vs
                            # avoid at the MBON rows, active KCs, top
                            # MBONs — measured on the SAME pass
                            try:
                                acts = h[self.mbon]
                                app_w = float(acts[self._mbon_approach].sum())
                                av_w = float(acts[self._mbon_avoid].sum())
                                bal = math.tanh(
                                    2.0 * (app_w - av_w)
                                    / (abs(app_w) + abs(av_w) + 1e-9))
                                top_idx = np.argsort(acts)[-3:][::-1]
                                names = getattr(self, "_mbon_names", None)
                                tops = []
                                for ti in top_idx:
                                    root = int(self.mbon[ti])
                                    nm = (names[ti] if names and ti < len(names)
                                          and names[ti] else None)
                                    tops.append({"mbon": nm or f"#{str(root)[-6:]}",
                                                 "h": round(float(acts[ti]), 5),
                                                 "nt": str(self.mbon_nt[ti])})
                                self._last_whole = {
                                    "balance": round(bal, 6),
                                    "approach": round(app_w, 6),
                                    "avoid": round(av_w, 6),
                                    "kc_active": int(np.count_nonzero(
                                        h[self.kc])),
                                    "hop_counts": hop_counts,
                                    "dn_drive": float(
                                        getattr(self, "last_conf", 0) or 0),
                                    "top_mbons": tops,
                                    "ms": self.last_pass_ms,
                                }
                            except Exception:
                                pass
                            self.last_dn_share = share
                            # confidence: activity reaching DNs relative to
                            # their population baseline, clamped 0.2..1.0
                            base = (float(self.dn_mask.sum())
                                    / max(1, len(self.dn_mask)))
                            rel = min(1.0, share / base) if base > 0 else 0.0
                            self.last_conf = round(0.2 + 0.8 * rel, 4)
                            self.last_pass_ms = round(
                                (time.perf_counter() - t0) * 1000, 1)
                            self.last_pass_at = time.time()
                            self.pass_age_s = 0.0
                except Exception:
                    pass
                self._stop.wait(WHOLE_PASS_PERIOD_S)

        self._thread = threading.Thread(target=loop, name="flywire-pass",
                                        daemon=True)
        self._thread.start()

    def _inject_whole(self, rates: np.ndarray) -> np.ndarray:
        h0 = np.zeros(len(self.roots), dtype=np.float32)
        alpn_g = self.alpn
        n_groups = len(rates)
        bounds = np.linspace(0, len(alpn_g), n_groups + 1).astype(int)
        for g in range(n_groups):
            seg = alpn_g[bounds[g]:bounds[g + 1]]
            if len(seg):
                h0[seg] = rates[g]
        return h0

    # ------------------------------------------------------ RPE learning
    def rpe_update(self, vote: float, direction: int, actual_move: float,
                   horizon: str = "tf1") -> dict | None:
        """Dopamine RPE only (§7): updates KC→MBON of ACTIVE KCs at
        window boundaries. Nothing else may change the vote's sign (B-10)."""
        if self.state != "ready" or direction == 0 or vote == 0:
            return None
        R = 1.0 if (1 if actual_move > 0 else -1) == direction else -1.0
        E_prev = self.E
        # EMA of recent confidence-weighted outcomes
        self.E = 0.8 * self.E + 0.2 * R
        delta = R - E_prev
        self.drg = math.tanh(RPE_LAMBDA * delta)
        self.alpha = float(np.clip(0.10 + 0.40 * abs(delta), 0.10, 0.50))
        self.phasic = self.phasic * RPE_PHASIC_DECAY + self.drg * 0.3
        # snapshot for rollback (hourly hindsight may revert us)
        self._rollback_snapshot = self._snapshot_rpe()
        # w += α · drg · pre · post  on active KC→MBON of the active KCs
        # (applied at boundaries only — this method is called at settle)
        entry = {"ts": time.time(), "horizon": horizon, "R": R,
                 "E": round(self.E, 4), "delta": round(delta, 4),
                 "drg": round(self.drg, 4),
                 "alpha": round(self.alpha, 4)}
        self.rpe_history.append(entry)
        self.rpe_history = self.rpe_history[-50:]
        return entry

    def apply_rpe_weights(self, kc_pre: np.ndarray, mbon_post: np.ndarray,
                          sign_by_mbon: np.ndarray) -> None:
        """Actual weight update (called with the active KC set)."""
        if self._kc_mbon_delta is None:
            return
        a = self.alpha * self.drg
        if abs(a) < 1e-9:
            return
        # outer product over active pairs, sign per MBON class
        for kc_i, mb_j, s in zip(kc_pre, mbon_post, sign_by_mbon):
            if 0 <= kc_i < self._kc_mbon_delta.shape[0] and \
                    0 <= mb_j < self._kc_mbon_delta.shape[1]:
                self._kc_mbon_delta[kc_i, mb_j] += np.float32(a * s)

    def _snapshot_rpe(self) -> dict:
        return {"E": self.E, "drg": self.drg, "phasic": self.phasic,
                "alpha": self.alpha,
                "delta": (self._kc_mbon_delta.copy()
                          if self._kc_mbon_delta is not None else None)}

    def rollback(self) -> bool:
        """Hindsight found the brain worse than before — revert the last
        RPE change (returns True if a snapshot existed)."""
        snap = self._rollback_snapshot
        if not snap:
            return False
        with self._lock:
            self.E = snap["E"]
            self.drg = snap["drg"]
            self.phasic = snap["phasic"]
            self.alpha = snap["alpha"]
            if snap["delta"] is not None and self._kc_mbon_delta is not None:
                self._kc_mbon_delta = snap["delta"]
            self._rollback_snapshot = None
        return True

    def state_dict(self) -> dict:
        return {
            "build_version": BUILD_VERSION,
            "E": self.E, "drg": self.drg, "phasic": self.phasic,
            "alpha": self.alpha,
            "flips": self.flips.tolist() if self.flips is not None else None,
            "rpe_history": self.rpe_history[-20:],
            "delta": (self._kc_mbon_delta.astype(np.float16).tolist()
                      if self._kc_mbon_delta is not None else None),
        }

    def load_state(self, d: dict) -> None:
        """Restore learners — constants (floors, BUILD_VERSION) always
        come from code, never from state (zeros-cause-3)."""
        if not d or int(d.get("build_version", -1)) != BUILD_VERSION:
            return
        if self.state != "ready" and (
                d.get("flips") is not None or d.get("delta") is not None):
            # matrices not loaded yet (async build) — apply after _load()
            self._pending_state = d
        self.E = float(d.get("E", 0.0))
        self.drg = float(d.get("drg", 0.0))
        self.phasic = float(d.get("phasic", 0.0))
        self.alpha = float(np.clip(float(d.get("alpha", 0.10)), 0.10, 0.50))
        self.rpe_history = list(d.get("rpe_history") or [])
        fl = d.get("flips")
        if fl is not None and self.flips is not None and \
                len(fl) == len(self.flips):
            self.flips = np.array(fl, dtype=np.int8)
        delta = d.get("delta")
        if delta is not None and self._kc_mbon_delta is not None:
            try:
                arr = np.array(delta, dtype=np.float32)
                if arr.shape == self._kc_mbon_delta.shape:
                    self._kc_mbon_delta = arr
            except Exception:
                pass

    # -------------------------------------------------------- formulas
    def ccsv2(self) -> dict:
        """Circuit consensus: tanh(LH_approach − LH_avoid) over the MB
        pass on the current input (signed brain formula)."""
        if self.state != "ready":
            return {"value": 0.0, "zero_reasons": "brain absent",
                    "kind": "signed"}
        values = list(self._last_values or [])
        if not values or not any(abs(v) > 1e-9 for v in values):
            return {"value": 0.0,
                    "zero_reasons": "flat input (no channel active)",
                    "kind": "signed"}
        r = self.pass_mb(self.encode(values))
        if not r.get("ok"):
            return {"value": 0.0,
                    "zero_reasons": r.get("reason", "brain absent"),
                    "kind": "signed"}
        # LH readout classified by its own neurotransmitter balance:
        # reuse approach/avoid from MBONs + LH activity as magnitude
        lh = float(r.get("lh", 0.0))
        signed = math.tanh((r["approach"] - r["avoid"])
                           * (1.0 + lh))
        return {"value": round(signed, 6), "kind": "signed",
                "zero_reasons": None,
                "trace": [{"label": "approach", "value": r["approach"]},
                          {"label": "avoid", "value": r["avoid"]},
                          {"label": "lh", "value": lh}]}

    def kcae(self) -> dict:
        """Kenyon-cell assembly entropy over the last 10 evaluations —
        low entropy = stable percept = higher confidence (gauge, never a
        signed direction; flat watch must ignore it — zeros cause 7)."""
        if self.state != "ready":
            return {"value": 0.0, "zero_reasons": "brain absent",
                    "kind": "gauge"}
        if len(self._kcae_ring) < 3:
            return {"value": 0.0,
                    "zero_reasons": (f"needs ≥3 evaluations, had "
                                     f"{len(self._kcae_ring)}"),
                    "kind": "gauge"}
        H = float(np.mean(self._kcae_ring))
        return {"value": round(1.0 - H, 6), "kind": "gauge",
                "zero_reasons": None,
                "trace": [{"label": "mean entropy (10)", "value": H},
                          {"label": "n", "value": len(self._kcae_ring)}]}

    def note_kc(self, kc_act: np.ndarray | None) -> None:
        if kc_act is None:
            return
        s = np.asarray(kc_act, dtype=np.float64)
        s = s[s > 0]
        if len(s) == 0:
            self._kcae_ring.append(1.0)          # uniform = max entropy
        else:
            p = s / s.sum()
            H = float(-(p * np.log(p + 1e-12)).sum())
            Hmax = math.log(len(s)) if len(s) > 1 else 1.0
            self._kcae_ring.append(min(1.0, H / Hmax))
        self._kcae_ring = self._kcae_ring[-10:]

    # ------------------------------------------------------ anatomy
    def anatomy(self) -> dict:
        """Full connectome census (user: 'full neurons, cells, synapses
        and others'): class histogram, cell_class × cell_class weight
        matrix, synapse total, NT/side/cell-type breakdowns.

        Cheap parts (class counts) answer instantly from the npz; the
        heavy passes (15 M-edge matrix, parquet/TSV reads) run ONCE in a
        background thread — the first call answers 'computing…'.
        """
        if self.state != "ready":
            return {"state": "absent"}
        if self._anatomy is not None:
            return {"state": "ready", **self._anatomy}
        if self._anatomy_state == "idle":
            self._anatomy_state = "computing"
            threading.Thread(target=self._build_anatomy, name="flywire-anatomy",
                             daemon=True).start()
        if self._anatomy_state == "error":
            return {"state": "error", "error": self._anatomy_error}
        # instant part while the heavy pass runs
        classes, counts = np.unique(self.cls, return_counts=True)
        return {"state": "computing",
                "neurons": int(len(self.roots)),
                "classes": {str(c): int(n)
                            for c, n in zip(classes, counts)}}

    def _build_anatomy(self) -> None:      # pragma: no cover - thread
        try:
            # ── class histogram (cheap, already in hand)
            classes, counts = np.unique(self.cls, return_counts=True)
            class_names = [str(c) for c in classes]
            k = len(class_names)
            cls_code = np.searchsorted(classes, self.cls).astype(np.int32)

            # ── cell_class × cell_class summed |W| (the brain.md §9
            #    block matrix) over ALL 15 M edges, chunked by row band
            W = self.W
            M = np.zeros((k, k), dtype=np.float64)
            M_signed = np.zeros((k, k), dtype=np.float64)
            indptr, indices, data = W.indptr, W.indices, W.data
            n = len(indptr) - 1
            band = 8192
            for r0 in range(0, n, band):
                r1 = min(n, r0 + band)
                rows = np.repeat(cls_code[r0:r1],
                                 np.diff(indptr[r0:r1 + 1]))
                cols = cls_code[indices[indptr[r0]:indptr[r1]]]
                w = data[indptr[r0]:indptr[r1]]
                M += np.bincount(rows * k + cols,
                                 weights=np.abs(w),
                                 minlength=k * k).reshape(k, k)
                # signed sum → excitatory (positive) vs inhibitory (negative)
                M_signed += np.bincount(rows * k + cols,
                                        weights=w,
                                        minlength=k * k).reshape(k, k)

            # ── formula-channel × population net drive (the old app's
            #    'brain matrix': one row per input channel, signed)
            n_ch = self.n_channels()
            M_ch = np.zeros((n_ch, k), dtype=np.float64)
            try:
                csc = W.tocsc()
                cp = csc.indptr
                bounds = np.linspace(0, len(self.alpn), 2 * n_ch + 1
                                     ).astype(int)
                for gi in range(2 * n_ch):
                    ch = gi if gi < n_ch else gi - n_ch
                    sign_side = 1.0 if gi < n_ch else -1.0
                    for a in self.alpn[bounds[gi]:bounds[gi + 1]]:
                        a = int(a)
                        seg_rows = csc.indices[cp[a]:cp[a + 1]]
                        seg_vals = csc.data[cp[a]:cp[a + 1]]
                        if not len(seg_rows):
                            continue
                        M_ch[ch] += sign_side * np.bincount(
                            cls_code[seg_rows], weights=seg_vals,
                            minlength=k)
                del csc
            except Exception:
                M_ch = np.zeros((n_ch, k), dtype=np.float64)

            # ── synapse total + annotation census (parquet/TSV, once)
            synapses = None
            top_types: dict[str, int] = {}
            sides: dict[str, int] = {}
            nt_census: dict[str, int] = {}
            try:
                import pyarrow.parquet as pq
                pq_path = getattr(self, "_parquet_path",
                                  CACHE_DIR / PARQUET)
                col = pq.read_table(pq_path,
                                    columns=["Connectivity"]).column(0)
                synapses = int(np.asarray(col.to_numpy(),
                                          dtype=np.int64).sum())
                del col
            except Exception:
                synapses = None      # honest: unknown, never invented
            try:
                import pyarrow.csv as pcsv
                tsv_path = getattr(self, "_ann_path",
                                   CACHE_DIR / ANNOTATIONS)
                hdr = open(tsv_path, encoding="utf-8").readline()
                have = hdr.rstrip("\n").split("\t")
                want = [c for c in ("cell_type", "side", "top_nt")
                        if c in have]
                tbl = pcsv.read_csv(
                    tsv_path,
                    parse_options=pcsv.ParseOptions(delimiter="\t"),
                    convert_options=pcsv.ConvertOptions(
                        include_columns=want))
                if "cell_type" in want:
                    tc: dict[str, int] = {}
                    for v in tbl.column("cell_type").to_pylist():
                        v = str(v or "?")
                        tc[v] = tc.get(v, 0) + 1
                    top_types = dict(sorted(tc.items(),
                                            key=lambda kv: -kv[1])[:14])
                if "side" in want:
                    sd: dict[str, int] = {}
                    for v in tbl.column("side").to_pylist():
                        v = str(v or "?")
                        sd[v] = sd.get(v, 0) + 1
                    sides = sd
                if "top_nt" in want:
                    nt: dict[str, int] = {}
                    for v in tbl.column("top_nt").to_pylist():
                        v = str(v or "?")
                        nt[v] = nt.get(v, 0) + 1
                    nt_census = dict(sorted(nt.items(),
                                            key=lambda kv: -kv[1]))
            except Exception:
                pass                # optional enrichment only

            # ── derived stats
            edges = int(W.nnz)
            neurons = int(len(self.roots))
            mbon_nt: dict[str, int] = {}
            for t in self.mbon_nt.tolist():
                mbon_nt[t] = mbon_nt.get(t, 0) + 1
            # strongest population → population pathways (signed |W| sum)
            pathways = []
            flat = [(i, j, float(M_signed[i, j]))
                    for i in range(k) for j in range(k)
                    if M_signed[i, j] != 0.0]
            flat.sort(key=lambda t: -abs(t[2]))
            for i, j, v in flat[:12]:
                pathways.append({
                    "frm": class_names[i], "to": class_names[j],
                    "drive": round(v, 4),
                    "sign": "excitatory" if v >= 0 else "inhibitory",
                })
            # MBON display names from the annotations (cell_type)
            mbon_names: list[str | None] = []
            try:
                import pyarrow.csv as pcsv2
                tsv2 = getattr(self, "_ann_path", CACHE_DIR / ANNOTATIONS)
                at = pcsv2.read_csv(
                    tsv2,
                    parse_options=pcsv2.ParseOptions(delimiter="\t"),
                    convert_options=pcsv2.ConvertOptions(
                        include_columns=["root_id", "cell_type"]))
                rts = at.column("root_id").to_numpy().astype(np.int64)
                cts = [str(v or "") for v in at.column("cell_type")]
                order2 = np.argsort(rts, kind="stable")
                rts, cts = rts[order2], [cts[i] for i in order2]
                pos2 = np.searchsorted(rts, self.mbon)
                pos2[pos2 >= len(rts)] = 0
                mbon_names = [
                    (cts[p] if rts[p] == int(root) and cts[p] else None)
                    for p, root in zip(pos2, self.mbon)
                ]
                self._mbon_names = mbon_names
            except Exception:
                mbon_names = [None] * len(self.mbon)

            self._anatomy = {
                "neurons": neurons,
                "connections": edges,
                "synapses": synapses,
                "mean_degree": round(edges / neurons, 2) if neurons else 0,
                "density": round(edges / (neurons * max(1, neurons - 1)), 6),
                "classes": {c: int(n) for c, n in
                            zip(class_names, counts.tolist())},
                "block": {"classes": class_names,
                          "matrix": [[round(float(v), 4) for v in row]
                                     for row in M]},
                "mbon_nt": mbon_nt,
                "top_cell_types": top_types,
                "sides": sides,
                "nt_census": nt_census,
                "block_signed": [[round(float(v), 5) for v in row]
                                 for row in M_signed],
                "pathways": pathways,
                "channel_matrix": {
                    "channels": int(n_ch),
                    "classes": class_names,
                    "matrix": [[round(float(v), 5) for v in row]
                               for row in M_ch],
                    "note": ("net drive of formula channel c into "
                             "population class (ON − OFF at x=+1)"),
                },
                "mbon_names": [m for m in mbon_names if m] or None,
                "built_at": time.time(),
            }
            self._anatomy_state = "ready"
        except Exception as exc:
            self._anatomy_state = "error"
            self._anatomy_error = repr(exc)

    # -------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "state": self.state,
            "progress": self.progress,
            "last_error": short_error(self.last_error),
            "verify": self.steps or self.verify(),
            "meta": self.meta,
            "chip": self.chip(),
            "pass": {"ms": self.last_pass_ms,
                     "age_s": (round(time.time() - self.last_pass_at, 1)
                               if getattr(self, "last_pass_at", 0) else None),
                     "dn_share": self.last_dn_share,
                     "conf": self.last_conf},
            "polarity": {"flips": (self.flips.tolist()
                                   if self.flips is not None else None),
                         "dead": self.dead_channels,
                         "n_channels": (self.n_channels()
                                        if self.state == "ready" else 0)},
            "dopamine": {"E": round(self.E, 4),
                         "delta": round(self.rpe_history[-1]["delta"], 4)
                         if self.rpe_history else 0.0,
                         "drg": round(self.drg, 4),
                         "alpha": round(self.alpha, 4),
                         "tonic": self.tonic,
                         "phasic": round(self.phasic, 4),
                         "history": self.rpe_history[-5:]},
            "formulas": {"ccsv2": self.ccsv2(), "kcae": self.kcae()},
            "anatomy": self.anatomy(),
            "readout": {"whole": self._last_whole, "mb": self._last_mb},
            "build_version": BUILD_VERSION,
        }
