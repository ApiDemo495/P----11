/* ═══════════════════════════════════════════════════════════════
   PREDICT — application (rewrite r2)
   One snapshot contract, one tick stream, nine views.
   ═══════════════════════════════════════════════════════════════ */

const TF = ["tf1", "tf2", "tf3", "tf4"];
const TF_META = {
  tf1: { mins: 22, tps: [5, 10, 15, 22] },
  tf2: { mins: 44, tps: [11, 22, 33, 44] },
  tf3: { mins: 130, tps: [30, 40, 90, 130] },
  tf4: { mins: 247, tps: [60, 120, 180, 247] },
};
const ASSETS = ["BTC", "PAXG", "ETH", "BNB"];
// genuine MARKET emotions, named for what each E-channel measures
// (Pride/Envy/Guilt were not market emotions — removed). Retail vs
// hedge crowd reads are computed server-side from these channels.
// 21 channels from published affect models (see app/emotion.py). The server
// sends the same list in em.model.channels — this is the offline fallback.
const EMO = ["Valence","Arousal","Dominance","Positive affect","Negative affect",
  "Fear","Anger","Disgust","Sadness","Surprise","Enjoyment",
  "Loss-aversion pressure","Somatic marker","Herding","Capitulation",
  "Extrapolation / chase","Volatility stress","Momentum vs mean",
  "Market breadth","News sentiment","Trend exhaustion","Net emotion"];
const P_NAMES = {
  1:"QTPB quantised price ladder",2:"WFCP wave-function collapse",3:"TME trend memory",
  4:"QTPB ladder density",5:"PSI phase state",6:"DRHE/DRPT trend decoherence",
  7:"Lambda critical burst",8:"PSI-2 phase drift",9:"TME×coupling",10:"Lambda-2",
  11:"PDI phase decoherence",12:"Cohesion",13:"Entanglement",14:"Tunnelling prob.",
  15:"Superposition width",16:"Observation collapse",17:"Decoherence rate",
  18:"Interference term",19:"GSPI gauge non-invariance",20:"Tunnelling cost",
  21:"QDPC correlation time",
};
const P_LOGIC = {
  1:"Quantised price ladder — P(x) over ATR-spaced levels → predicted levels",
  2:"Wave-function collapse probability of the price state → confidence ×(0.5+0.5·WFCP)",
  3:"Trend memory: how long the current drift has persisted",
  4:"Ladder density — concentration of probability mass on nearby levels",
  5:"Phase state of the cycle (ψ components)",
  6:"Trend decoherence DRPT — kills confidence ×(1−DRPT) in chop",
  7:"λ-critical: burst-probability when autocorrelation spikes",
  8:"Phase drift of the order-flow state",
  9:"TME × Bgn coupling term",
  10:"Secondary criticality (λ₂)",
  11:"Phase decoherence index",
  12:"Cross-window cohesion of the probability ladder",
  13:"Entanglement of tf↔tf signals (Bgn β=1.2)",
  14:"Tunnelling probability through the resistance level",
  15:"Superposition width σ_q of the predicted price",
  16:"Observation collapse: how observation (volume) narrows σ",
  17:"Decoherence rate of the quantum state",
  18:"Interference term between up/down branches",
  19:"GSPI gauge non-invariance — final confidence gate ×(1−GSPI)",
  20:"Tunnelling cost (barrier thickness)",
  21:"QDPC correlation time τ of the price state",
};
const FORMULA_LOGIC = {
  1:"EMA-fast vs EMA-slow separation / ATR → trend push strength",
  2:"rolling return z-score vs baseline σ → stretched-move detector",
  3:"higher-high / lower-low structure count → regime",
  4:"volume-weighted momentum Σ(vol×Δp)/ATR → conviction",
  5:"distance of price from VWAP band, signed → mean-reversion pull",
  6:"RSI-style oscillator mapped to [−1,+1]",
  7:"consecutive same-direction closes → exhaustion risk",
  8:"fractal pivot stack swing lows−highs → pressure",
  9:"rescaled-range of the window → trend persistence",
  10:"ATR expansion ratio now/baseline → volatility regime shift",
  11:"buy-vs-sell volume delta at extremes → aggressor side",
  12:"gap fill probability vs prior range midpoint",
  13:"time-of-day seasonal drift component",
  14:"cross-asset lead PAXG vs BTC residual direction",
  15:"drawdown depth/speed from window high → capitulation",
  16:"compression of true range → squeeze breakout potential",
  17:"imbalance of large-wick rejections → rejection strength",
  18:"entropy of return signs → trendiness",
  19:"price acceleration ΔΔclose → momentum of momentum",
  20:"DRHE trend durability — confidence gate ×min(1,DRHE)",
  21:"CCI master aggregate: 45 weighted components (F1-F44 + 6 subsystems)",
  22:"breakout age since level break → early/late",
  23:"round-number liquidity cluster pressure",
  24:"session-open positioning vs prior close",
  25:"funding-proxy from consecutive impulse sizes",
  26:"relative strength vs Bgn-synced self (cross-TF)",
  27:"tail-risk kernel: extreme-move probability",
  28:"ARC confidence/volatility ratio band",
  29:"DLP price LEVELS (not a direction)",
  30:"PCV post-CCI conflict validator",
  31:"final blend gate into the CCI stack",
};

/* ── tiny helpers ───────────────────────────────────────────── */
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
// §10: escapeHtml everything from the server — one helper, used everywhere
function esc(x) {
  return String(x ?? "").replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// tiny non-zero values must not read as "0.00" — the user reads that as
// "the formula is not being used" (honesty: show the actual magnitude)
const fmtSig = (v) => {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (n === 0) return "0.000";
  if (Math.abs(n) < 0.001) return n.toPrecision(2);
  return n.toFixed(3);
};

/* zeros-cause-8: a value that is small but REAL must print its first
   significant digit — up to 6 decimals, then exponent. True zero prints
   0.00. Use for every human-facing near-zero. */
const ZCAUSE = {
  error: "✗ error", equal: "＝ both sides equal", input: "input missing",
  history: "needs history", materiality: "materiality gate",
  flat: "computed flat",
};
function fx(v, d = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  if (!isFinite(v)) return v > 0 ? "+∞" : "−∞";
  const a = Math.abs(v);
  if (a === 0) return (0).toFixed(d);
  if (a >= 10 ** -d) return v.toFixed(d);
  for (let k = d + 1; k <= 6; k++) {
    const t = v.toFixed(k);
    if (Math.abs(parseFloat(t)) > 0) return t;
  }
  return v.toExponential(2);
}
const fmt = (v, d = 2) => v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d);
/* A missing number is never a measured zero. `|| 0` / `?? 0` written before
   toFixed() silently turns "we have no reading" into "the reading is 0.00",
   which is how the book panel once claimed a 0% win rate and PF 0.00 for
   arms that had not traded yet. num() is the only sanctioned path. */
const num = (v, d = 2) => {
  if (v == null) return "—";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : "—";
};
const px = v => v >= 1000 ? Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 })
  : v >= 10 ? Number(v).toFixed(2) : Number(v).toFixed(4);
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg;
  $("#toast-wrap").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}
async function api(path, opts) {
  const ctl = new AbortController();
  const to = setTimeout(() => ctl.abort(), 10000);
  try {
    const r = await fetch(path, { ...(opts || {}), signal: ctl.signal });
    if (!r.ok) {
      // keep the server's JSON body (cell_locked countdowns, refusal
      // reasons) reachable for callers — the status line alone loses it
      let body = null;
      try { body = await r.json(); } catch { /* non-JSON error */ }
      const err = new Error((body && (body.why || body.error))
        ? String(body.why || body.error) : `${path} → ${r.status}`);
      err.status = r.status;
      err.body = body;
      throw err;
    }
    return await r.json();
  } catch (e) {
    // A timeout means the engine is BUSY, not absent. The engine's event loop
    // legitimately stalls for minutes during a warm replay or a GRC pass, and
    // the old code let those aborts fall through to the offline banner, which
    // then told the user to "start it: bash run.sh" while it was running.
    if (e && e.name === "AbortError") {
      const err = new Error(`${path} → timed out after 10 s (engine busy)`);
      err.busy = true;
      throw err;
    }
    throw e;
  } finally { clearTimeout(to); }
}

/* ── state ──────────────────────────────────────────────────── */
const S = {
  symbol: "BTC", focusTf: "tf1", tfFocus: false, view: "dashboard",
  snap: null, prices: {}, lastMsgAt: 0, wsRetry: 0,
  cycles: {}, grcFlash: {}, lastSec: {},
  charts: {}, spark: {},
  newsSel: "BTC", histDay: null,
};

/* ═══════════ shell: assets, tabs, focus bar ═══════════ */
document.addEventListener("DOMContentLoaded", () => {
  $("#btn-repair-link")?.addEventListener("click", () => repairChartLink(true));
  $("#btn-system")?.addEventListener("click", async () => {
    const next = S.systemMode === "new" ? "old" : "new";
    const b = $("#btn-system");
    if (b) b.disabled = true;
    try {
      const r = await api("/api/system/mode", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: next }) });
      if (r?.ok) {
        S.systemMode = r.system_mode || next;
        syncSystemBtn();
        // retarget the OPEN cycles immediately (server already did); the
        // snapshot below repaints every card from one source of truth
        try { renderHero(); renderQuads(); } catch (e) { /* optional */ }
        api("/api/snapshot").then(applySnapshot).catch(() => {});
        toast(`prediction model: ${S.systemMode.toUpperCase()} · ${r.cycles_retargeted ?? 0} windows retargeted`, "ok");
      } else {
        toast("system switch refused", "bad");
      }
    } catch (e) { toast("system switch failed: " + (e.message || e), "bad"); }
    finally { if (b) b.disabled = false; }
  });
  $("#boot-dismiss")?.addEventListener("click", () => {
    S.bootDismissed = true;
    $("#boot-overlay")?.classList.add("gone");
    $("#boot-chip")?.classList.remove("hidden");
  });
  $("#boot-chip")?.addEventListener("click", () => {
    S.bootDismissed = false;
    $("#boot-chip")?.classList.add("hidden");
    $("#boot-overlay")?.classList.remove("gone");
  });
});
function buildRail() {
  const list = $("#asset-list"); list.innerHTML = "";
  ASSETS.forEach(sym => {
    const b = document.createElement("button");
    b.className = "asset-btn" + (sym === S.symbol ? " active" : "");
    b.dataset.sym = sym;
    b.innerHTML = `<span class="sym">${sym}</span><span class="px" id="ap-${sym}">—</span>
      <span class="chg" id="ac-${sym}"></span>`;
    b.addEventListener("click", () => setSymbol(sym));
    list.appendChild(b);
  });
}
function setSymbol(sym) {
  S.lastViewFetch = 0;
  const ab = $("#analyse-sym"); if (ab) ab.textContent = sym;
  if (sym !== S.symbol) {
    // Invalidate any request for the previous asset and never let its late
    // response populate the newly selected asset's candle buffers.
    S._candleLoadSeq = (S._candleLoadSeq || 0) + 1;
    S.tfBars = {}; S.tfBarsRevision = {}; S.tfBarsSymbol = sym;
    S.tfBarsForeign = 0; S.tfBarsIntegrity = null; S.tfBarsFetchedAt = 0;
    S.tfBarsFamily = "unknown"; S.tfBarsMode = "unknown";
    S.candleSeries = null; S.priceHist = [];
    resetOwnChartForSymbol(sym);
  }
  S.symbol = sym;
  try { localStorage.setItem("symbol", sym); } catch (e) { /* private */ }
  $$(".asset-btn").forEach(b => b.classList.toggle("active", b.dataset.sym === sym));
  syncToolbar();
  // ATOMIC SWITCH (defect: the hero kept BTC's locked TP/SL, its
  // "ANALYSING …" card and the sub-timeframe strip until the next
  // snapshot landed). Wipe every asset-bound surface NOW, so nothing can
  // show the previous asset's numbers under the new ticker; the first
  // fresh snapshot/MODE card repaints them.
  clearAssetSurfaces(sym);
  refreshActiveView(true);
  buildQuads(); loadCandles();
}

/* Blank/park every surface that is bound to ONE asset. Called on an asset
   switch, before the new snapshot is applied. */
function clearAssetSurfaces(sym) {
  const ttl = $("#hero-title");
  if (ttl) ttl.innerHTML = `${S.focusTf.toUpperCase()} · <b style="color:var(--cyan)">${TF_META[S.focusTf].mins}m</b> window`
    + ` <span class="dim">${sym} · loading the ${sym} tape…</span>`;
  const badge = $("#hero-badge");
  if (badge) { badge.className = "hero-badge flat"; badge.innerHTML = `● ANALYSING ${sym}…<small>${sym} · ${S.focusTf.toUpperCase()} — waiting for the first ${sym} snapshot</small>`; }
  const stats = $("#hero-stats");
  if (stats) stats.innerHTML = `<div class="kv"><span>call</span><b class="dim">loading ${sym}…</b></div>`;
  const strip = $("#subtf-strip");
  if (strip) strip.innerHTML = `<div class="dim pad">waiting for the first ${sym} lock…</div>`;
  const sl = $("#hero-sleeve");
  if (sl) sl.innerHTML = `<div class="kv"><span>certified sleeve</span><b class="dim">loading ${sym}…</b></div>`;
  const rn = $("#risk-note");
  if (rn) rn.textContent = `${sym}: TP/SL come from the measured-moves block of the first ${sym} lock — showing nothing until then.`;
  const ap = $("#analyse-panel");
  if (ap) ap.classList.add("hidden");   // previous asset's ANALYSING card
  const anT = $("#an-title");
  if (anT) anT.textContent = `ANALYSING ${sym} · ${S.focusTf.toUpperCase()}`;
  // charts: drop the old asset's candles + overlays immediately
  TF.forEach(tf => {
    try { S.charts[tf]?.setData([]); S.charts[tf]?.setLines([]); } catch (e) { /* not built yet */ }
    try { S.charts[tf]?.tick?.(0); } catch (e) { /* noop */ }
  });
  if (typeof Bn !== "undefined") {
    if (Bn.main) { try { Bn.main.candles.setData([]); } catch (e) {} Bn.main.last = null; }
    Object.values(Bn.quads || {}).forEach(inst => { try { inst.candles.setData([]); } catch (e) {} inst.last = null; });
  }
}
/* ═══════ EXCHANGE PHYSICS panel (Chrono-Sieve-inspired, measured only) ═ */
function renderMicro() {
  const box = $("#micro-panel");
  if (!box) return;
  const sys = S.snap?.systems?.[`${S.symbol}|${S.focusTf}`] || {};
  const m = sys.micro;
  const bk = sys.book || {};
  $("#micro-meta").textContent = `${S.symbol} · ${m?.mode || "—"}`;
  if (!m && !bk.book) {
    box.innerHTML = `<div class="dim pad">${S.snap
      ? "engine pass pending — systems publish on the next GRC"
      : "loading…"}</div>`;
    return;
  }
  if (!m?.available && !bk.book) {
    box.innerHTML = `
      <div class="hint" style="color:var(--gold)">○ ${esc(m?.reason || "no exchange micro data")}</div>
      <div class="hint">gauges turn on with the live Binance feed: depth20 book (DBI, micro-price, Λ cancel elasticity, walls), τ = E−T clock-skew z, and @forceOrder liquidation clusters — measured prints only, no synthetic particles.</div>`;
    return;
  }
  // ---- clock-skew / liquidation block ----
  let microBlock;
  if (!m?.available) {
    microBlock = `<div class="hint" style="color:var(--gold)">○ clock-skew / forceOrder: ${esc(m?.reason || "unavailable")}</div>`;
  } else {
    if (!S.microHist) S.microHist = [];
    if (m.stress != null) S.microHist.push(m.stress);
    S.microHist = S.microHist.slice(-60);
    const spark = S.microHist.length > 2
      ? `<svg viewBox="0 0 100 26" preserveAspectRatio="none" style="width:120px;height:24px;vertical-align:middle"><polyline points="${
          S.microHist.map((v2, i) =>
            `${(i / Math.max(1, S.microHist.length - 1)) * 100},${24 - v2 * 22}`).join(" ")}"
          fill="none" stroke="#00d4f0" stroke-width="1.5"/></svg>` : "";
    const zline = (v2, n2, name2) => v2 == null
      ? `<div class="kv"><span>${name2}</span><b class="dim">needs ${m.min_samples} packets (had ${n2})</b></div>`
      : `<div class="kv"><span>${name2}</span><b class="${Math.abs(v2) >= 3 ? "acc-warn" : ""}">z ${v2}${Math.abs(v2) >= 3 ? " ⚠>3" : ""} <span class="dim">n=${n2}</span></b></div>`;
    const clu = (c2, arrow) => c2
      ? `<div class="kv"><span>cluster ${arrow}</span><b>${px(c2.price)} <span class="dim">${c2.bps} bps · ${c2.bias} · mass ${c2.mass}</span></b></div>`
      : `<div class="kv"><span>cluster ${arrow}</span><b class="dim">none ≥8% of flow</b></div>`;
    microBlock = `
      <div class="kv-row">
        <div class="kv" style="min-width:180px"><span>queue stress (τ, Δa)</span>
          <div class="bar" style="margin-top:4px"><i class="${(m.stress || 0) > 0.5 ? "neg" : "pos"}" style="width:${Math.max(2, Math.round((m.stress || 0) * 100))}%"></i></div>
          <b>${m.stress != null ? Math.round(m.stress * 100) + "%" : "—"}</b></div>
        ${zline(m.z_tau, m.n_tau, "z clock-skew (E−T, ms)")}
        ${zline(m.z_gap, m.n_gap, "z trade-id gap (Δa)")}
        <div class="kv"><span>net forced flow (session-decayed)</span><b class="${(m.net_flow || 0) >= 0 ? "acc-up" : "acc-dn"}">${m.net_flow != null ? (m.net_flow >= 0 ? "+" : "") + Math.round(m.net_flow * 100) + "% BUY-share" : "—"}</b></div>
        <div class="kv"><span>forceOrder prints</span><b>${m.liq_events} <span class="dim">${spark ? "· stress trace" : ""}</span></b></div>
      </div>
      <div class="kv-row">${clu(m.nearest_up, "↑ above")}${clu(m.nearest_dn, "↓ below")}</div>
      <div class="hint">stress = |z| of this box's OWN60-s measurement baseline (ms-level instrumentation — not a trading window) · clusters keep a full intraday session:6 h horizon,2 h half-life over real @forceOrder prints (25 bp buckets).</div>`;
  }
  // ---- live order book block (DepthBook) ----
  const bookBlock = bk.book ? `
    <div class="kv-row" style="margin-top:6px">
      <div class="kv"><span>depth imbalance DBI20</span><b class="${(bk.dbi || 0) >= 0 ? "acc-up" : "acc-dn"}">${bk.dbi != null ? (bk.dbi >= 0 ? "+" : "") + fx(bk.dbi, 3) : "—"}</b></div>
      <div class="kv"><span>micro-price pos (spread)</span><b class="${(bk.micro_pos || 0) >= 0 ? "acc-up" : "acc-dn"}">${bk.micro_pos != null ? (bk.micro_pos >= 0 ? "+" : "") + fx(bk.micro_pos, 3) : "—"}</b></div>
      <div class="kv"><span>spread</span><b>${bk.spread_bps != null ? bk.spread_bps + " bp" : "—"} <span class="dim">bid ${px(bk.bid)} × ask ${px(bk.ask)}</span></b></div>
      <div class="kv"><span>cancel elasticity Λ</span><b class="${Math.abs(bk.lambda || 0) >= 1.8 ? "acc-warn" : ""}">${bk.lambda != null ? fx(bk.lambda, 3) : "—"} <span class="dim">${esc(bk.lambda_reason || "")}</span></b></div>
      <div class="kv"><span>snapshots</span><b>${bk.snapshots} <span class="dim">${bk.age_s}s ago</span></b></div>
    </div>
    <div class="kv-row">
      ${bk.wall_bid ? `<div class="kv"><span>wall bid</span><b>${px(bk.wall_bid.price)} <span class="dim">${bk.wall_bid.bps}bp · qty ${bk.wall_bid.qty}</span></b></div>` : ""}
      ${bk.wall_ask ? `<div class="kv"><span>wall ask</span><b>${px(bk.wall_ask.price)} <span class="dim">${bk.wall_ask.bps}bp · qty ${bk.wall_ask.qty}</span></b></div>` : ""}
    </div>
    <div class="hint">DBI/micro/Λ measured from the live top-20 book (100 ms snapshots); Λ>+1.8 = makers pulling asks ~6× faster than bids (bullish vacuum), <−1.8 the reverse — feeds F34 as a weighted, hindsight-graded signal.</div>`
    : `<div class="hint" style="color:var(--gold)">○ ${esc(bk.reason || "no live order book yet")}</div>`;
  box.innerHTML = microBlock + bookBlock;
}

/* ═══════ SYSTEM panel (old-app inspiration): one glance at what runs ═ */
async function loadSystemPanel() {
  const box = $("#sys-panel"); if (!box) return;
  try {
    const d = await api("/api/system/status");
    $("#sys-meta").textContent =
      new Date((d.ts || Date.now() / 1000) * 1000).toLocaleTimeString();
    const cls = st => st === "ok" ? "acc-up"
      : st === "warn" ? "acc-warn"
      : st === "err" ? "acc-dn" : "dim";
    box.innerHTML = (d.rows || []).map(r =>
      `<div class="kv"><span>${esc(r.label)}</span>` +
      `<b class="${cls(r.state)}" title="${esc(r.detail)}">${esc(r.detail)}</b></div>`
    ).join("") || `<div class="dim pad">no status</div>`;
  } catch (e) {
    box.innerHTML = `<div class="dim pad">status: ${esc(e.message)}</div>`;
  }
}

/* ═══════ PROBABILITY BRANCHES chart (own canvas — no chart lib) ═══════ */
/* ═══ HERO: proper LINE CHART FROM NOW ═══
   Left: the realised price line up to a gold NOW divider. Right: the
   forward fan for the rest of the window (shaded q05–q95 / q25–q75
   cone + median), with TP/SL as future barriers and the locked entry
   across the whole chart. The last point of the line IS the live
   price — the chart can never lag the tape the way the 260-candle
   view did. */
/* Realised series for [t0, now]: candle closes (full-window backfill)
   merged with live ticks after the last bar. Shared by the hero and
   the probability chart so both lines start at the window start. */
/* Forward-table reading. Mirrors app/forward.py::estimate exactly:
   exact bucket at n>=30, else pooled +/-0.5 then +/-1.0, else a
   Laplace-smoothed frequency from whatever samples exist. Returns null
   only when the table has no samples at all. */
function fwdEstimate(cells, value) {
  const B = [-1, -0.5, 0, 0.5, 1];
  if (!cells || !cells.length) return null;
  const at = new Map(cells.map(c => [Number(c.bucket), c]));
  const b = B.reduce((a, c) => Math.abs(c - value) < Math.abs(a - value) ? c : a, B[0]);
  const exact = at.get(b);
  if (exact && exact.n >= 30) {
    return { p: exact.up != null ? exact.up / exact.n : exact.p_up,
             n: exact.n, span: 0, exact: true, bucket: b };
  }
  let fallback = null;
  for (const span of [0.5, 1.0]) {
    let up = 0, n = 0;
    for (const bb of B) {
      if (Math.abs(bb - b) > span + 1e-9) continue;
      const c = at.get(bb);
      if (c) { up += (c.up != null ? c.up : Math.round((c.p_up || 0) * c.n)); n += c.n; }
    }
    if (!n) continue;
    if (n >= 30) return { p: up / n, n, span, exact: false, bucket: b };
    fallback = { p: (up + 1) / (n + 2), n, span, exact: false,
                 smoothed: true, bucket: b };
  }
  if (exact && exact.n > 0) {
    return { p: (exact.up + 1) / (exact.n + 2), n: exact.n, span: 0,
             exact: false, smoothed: true, bucket: b };
  }
  return fallback;
}

function serverNow() { return Date.now() / 1000 + (S.serverOff || 0); }

function realisedSeries(t0, now) {
  const cs = (S.candleSeries && S.candleSeries.symbol === S.symbol)
    ? S.candleSeries.bars : [];
  let pts = cs.filter(p => p.t >= t0 - 60 && p.t <= now && p.p > 0);
  const lastBarT = pts.length ? pts[pts.length - 1].t : 0;
  const ticks = (S.priceHist || [])
    .filter(p => p.t > lastBarT && p.t >= t0 && p.t <= now && p.p > 0);
  pts = [...pts, ...ticks];
  // cut a stale seed/tick junction (>2% in <2 min) instead of a step
  for (let i = pts.length - 1; i > 0; i--) {
    if (Math.abs(pts[i].p / pts[i - 1].p - 1) > 0.02
        && pts[i].t - pts[i - 1].t < 120) { pts = pts.slice(i); break; }
  }
  return pts;
}

let _mainRaf = 0;
function drawMainChart() {
  if (S.chartSrc === "binance") { bnTickMain(); return; }
  if (S.chartSrc !== "own") return;
  const host = $("#chart-own"); if (!host) return;
  if (_mainRaf) return;
  _mainRaf = requestAnimationFrame(() => { _mainRaf = 0; _drawOwn(); });
}
function _drawOwn() { drawOwnChart(); }

/* ═══════ OWN chart (docs11-charts Part C): candlesticks from the
   server tape — the SAME stream that settles the levels — drawn with
   the vendored Lightweight Charts. Autoscale comes from the candles
   only; entry/TP/SL are price lines with axis tags (off-screen levels
   keep their tag at the axis edge, per the library); lock/hit markers;
   model q50 median estimate inside the current window (always on),
   optional q25/q75 range bands behind the FAN toggle; volume histogram;
   OHLC crosshair legend; stale-tape
   note on the chart itself (doc A-4). ═══════ */
const Own = { inst: null, key: "", lineKey: "", markerKey: "", fanKey: "__unset", symbol: "" };
const TZ_OFF = 19800;                    // IST = UTC+5:30 (doc D-2)
const lt = t => t + TZ_OFF;              // render-only shift for LWC
function resetOwnChartForSymbol(symbol) {
  if (Own.symbol === symbol) return;
  Own.symbol = symbol;
  Own.key = Own.rangeKey = Own.lineKey = Own.markerKey = "";
  Own.fanKey = "__unset";
  Own.rangeTo = 0;
  const inst = Own.inst;
  if (!inst) return;
  for (const line of Object.values(inst.lines || {})) {
    try { inst.candles.removePriceLine(line); } catch {}
  }
  inst.lines = {};
  inst.levelValues.splice(0, inst.levelValues.length);
  inst.last = null;
  try {
    inst.candles.setData([]); inst.vol.setData([]);
    inst.fan50.setData([]); inst.fan25.setData([]); inst.fan75.setData([]);
  } catch {}
  inst.forecastLabel.textContent = "forecast warming up";
  inst.forecastLabel.title = inst.forecastLabel.textContent;
  inst.stale.textContent = `loading ${symbol} candles…`;
}
/* TP/SL price lines must never flatten the candles. The old provider
   widened the axis to contain EVERY level, so a stop 1.9 % away squashed
   a 0.05 % intrabar range into one flat horizontal squiggle (audit §3.3).
   The candle range keeps at least 75 % of the visible height: a level may
   expand the axis by at most `maxExpand` (25 %) of the candle span, and a
   level beyond that is clipped off-scale (LWC keeps its axis tag). */
function levelAutoscaleProvider(values, maxExpand = 0.25) {
  return original => {
    const base = original?.();
    if (!base?.priceRange) return base;
    const finite = (values || []).filter(Number.isFinite);
    if (!finite.length) return base;
    const lo0 = base.priceRange.minValue, hi0 = base.priceRange.maxValue;
    const span = Math.max(1e-9, hi0 - lo0);
    const pad = span * Math.max(0, maxExpand);
    const lo = Math.min(lo0, Math.max(Math.min(...finite), lo0 - pad));
    const hi = Math.max(hi0, Math.min(Math.max(...finite), hi0 + pad));
    return { ...base, priceRange: { minValue: lo, maxValue: hi } };
  };
}
function ownMake(el) {
  const LW = window.LightweightCharts; if (!LW || !el) return null;
  // measure AFTER layout: a chart built while its container was still
  // hidden kept a 300 px canvas inside a 1200 px panel — the reported
  // "candles squeezed into the left 15%" of the OWN chart.
  const rect0 = el.getBoundingClientRect();
  const w0 = Math.max(320, Math.round(rect0.width || el.clientWidth || 0));
  const h0 = Math.max(240, Math.round(rect0.height || el.clientHeight || 340));
  const chart = LW.createChart(el, {
    layout: { background: { color: "transparent" }, textColor: "#cbd5e1", fontSize: 10 },
    grid: { vertLines: { color: "rgba(140,140,170,.07)" }, horzLines: { color: "rgba(140,140,170,.07)" } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 6 },
    rightPriceScale: { borderColor: "rgba(140,140,170,.25)", scaleMargins: { top: 0.08, bottom: 0.22 } },
    crosshair: { mode: LW.CrosshairMode.Normal },
    width: w0, height: h0,
  });
  const levelValues = [];
  const candles = chart.addCandlestickSeries({ upColor: "#26a69a", downColor: "#ef5350",
    borderVisible: false, wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    autoscaleInfoProvider: levelAutoscaleProvider(levelValues),
    priceFormat: { type: "price", precision: 2, minMove: 0.01 } });
  const vol = chart.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "volume" } });
  chart.priceScale("").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
  const fanOpts = { lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
  const fan50 = chart.addLineSeries({ ...fanOpts, lineWidth: 2, color: "#7fd8cf", lineStyle: LW.LineStyle.Dashed });
  const fan25 = chart.addLineSeries({ ...fanOpts, color: "rgba(38,166,154,.35)" });
  const fan75 = chart.addLineSeries({ ...fanOpts, color: "rgba(38,166,154,.35)" });
  const forecastLabel = document.createElement("div");
  forecastLabel.style.cssText = "position:absolute;top:6px;right:8px;z-index:5;max-width:calc(100% - 16px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:9px ui-monospace,monospace;color:#7fd8cf;background:rgba(10,14,20,.72);padding:2px 4px;border-radius:3px;pointer-events:auto;cursor:help";
  forecastLabel.textContent = "forecast warming up";
  forecastLabel.title = forecastLabel.textContent;
  el.appendChild(forecastLabel);
  const legend = document.createElement("div");
  legend.style.cssText = "position:absolute;top:6px;left:8px;z-index:5;font:10px ui-monospace,monospace;color:#c9c9dc;pointer-events:none";
  el.appendChild(legend);
  const stale = document.createElement("div");
  stale.style.cssText = "display:none;position:absolute;top:22px;left:8px;z-index:5;max-width:calc(100% - 16px);padding:4px 6px;border:1px solid rgba(239,83,80,.55);border-radius:4px;background:rgba(35,12,18,.86);font:10px ui-monospace,monospace;color:#ff9a9a;white-space:normal;pointer-events:none";
  el.appendChild(stale);
  chart.subscribeCrosshairMove(param => {
    const bar = param?.seriesData?.get(candles);
    legend.textContent = bar
      ? `O ${px(bar.open)} H ${px(bar.high)} L ${px(bar.low)} C ${px(bar.close)}`
      : "";
  });
  const refit = () => {
    const r = el.getBoundingClientRect();
    const w = Math.round(r.width || 0), h = Math.round(r.height || 0);
    if (w > 40 && h > 40) { try { chart.applyOptions({ width: w, height: h }); } catch {} }
  };
  new ResizeObserver(refit).observe(el);
  requestAnimationFrame(refit);
  setTimeout(refit, 250);
  window.addEventListener("resize", refit);
  return { chart, candles, vol, fan50, fan25, fan75, forecastLabel, legend, stale,
    lines: {}, levelValues, last: null };
}
function ownSetLines(inst, d) {
  const want = {};
  if (S.showLines !== false && d.levels_ok !== false) {
    if (Number.isFinite(d.entry)) want.entry = { price: d.entry, color: "#c9d1d9", title: "ENTRY", style: 1, contract: true };
    if (Number.isFinite(d.tp)) want.tp = { price: d.tp, color: "#26a69a", title: "TP", style: 2, contract: true };
    if (Number.isFinite(d.sl)) want.sl = { price: d.sl, color: "#ef5350", title: "SL", style: 2, contract: true };
  }
  const key = JSON.stringify(want);
  inst.levelValues.splice(0, inst.levelValues.length,
    ...Object.values(want).filter(v => v.contract).map(v => v.price));
  if (key === Own.lineKey) return;
  Own.lineKey = key;
  for (const k of Object.keys(inst.lines)) { try { inst.candles.removePriceLine(inst.lines[k]); } catch {} }
  inst.lines = {};
  for (const [k, v] of Object.entries(want))
    inst.lines[k] = inst.candles.createPriceLine({ price: v.price, color: v.color,
      lineWidth: 1, lineStyle: v.style, axisLabelVisible: true, title: v.title });
  try { inst.chart.priceScale("right").applyOptions({ autoScale: true }); } catch {}
}
function drawOwnChart() {
  const host = $("#chart-own"); if (!host) return;
  if (!Own.inst) { host.innerHTML = ""; Own.inst = ownMake(host); }
  const inst = Own.inst; if (!inst) return;
  resetOwnChartForSymbol(S.symbol); // clear the prior asset before reading its new tape
  const a = S.snap?.assets?.[S.symbol]; if (!a) return;
  const d = a.tfs?.[S.focusTf]; if (!d) return;
  const now = serverNow();
  // Fail closed on the exact rendered tape, not only a server-side link
  // sample. A stale chart must never sit below a current quote with no label.
  const age = a.feed_age_s;
  const bucket0 = (S.focusTf === "tf3" || S.focusTf === "tf4") ? 300 : 60;
  const bt0 = Math.floor(now / bucket0) * bucket0;
  const tapeArr = S.tfBars?.[S.focusTf] || [];
  const tapeLast = tapeArr[tapeArr.length - 1];
  // the 1 Hz tick stream is the live price truth; snapshot is fallback only.
  const liveP = (S.prices?.[S.symbol] > 0) ? S.prices[S.symbol] : a.price;
  const desync = (tapeLast && liveP > 0)
    ? Math.abs(tapeLast.close - liveP) / liveP : Infinity;
  const stalled = tapeLast && (now - tapeLast.time) > 2 * bucket0;
  const reasons = [];
  const family = S.tfBarsFamily || "unknown";
  const mode = S.tfBarsMode || "unknown";
  const integrity = S.tfBarsIntegrity;
  if (!(family === "binance" || family === "sim") || family !== mode)
    reasons.push("unknown/mismatched feed provenance");
  if (!integrity || integrity.ok !== true)
    reasons.push(integrity?.reasons?.length
      ? integrity.reasons.join(", ") : "server integrity unavailable");
  if (!S.tfBarsFetchedAt || Date.now() - S.tfBarsFetchedAt > 60000)
    reasons.push("candle response stale or unavailable");
  if (S.tfBarsForeign > 0)
    reasons.push(`${S.tfBarsForeign} foreign/unknown server candles`);
  if (!tapeLast) reasons.push("no active-source candles");
  if (!(liveP > 0)) reasons.push("live quote unavailable");
  if (tapeLast && liveP > 0 && desync * 10000 > 20)
    reasons.push(`quote/candle mismatch ${Math.round(desync * 10000)} bps`);
  if (tapeLast && (now - tapeLast.time) > Math.max(180, 2 * bucket0))
    reasons.push(`last candle ${Math.round(now - tapeLast.time)}s old`);
  if (desync > 0.002 || stalled) {
    if (Date.now() - (S._desyncAt || 0) > 5000) {   // rebuild at most 1/5s
      S._desyncAt = Date.now(); Own.key = "";
      loadCandles().catch(() => {});
    }
  }
  const notes = [];
  if (S.snap?.feed?.simulated) notes.push("SIM TAPE — simulated feed, not exchange bars"
    + (S.snap.feed.real_reason ? ` (${S.snap.feed.real_reason})` : ""));
  if (reasons.length)
    notes.push(`UNTRUSTED TAPE — ${reasons.join("; ")}; candles hidden`);
  if (desync > 0.002)
    notes.push(`CHART DESYNC ${(desync * 100).toFixed(1)}% — candles hidden, refetching`);
  if (stalled) notes.push("live bar stalled — refetching");
  if (!reasons.length && !stalled && age != null && age > 15)
    notes.push(`stale — last print ${Math.round(age)}s ago`);
  inst.stale.textContent = notes.join(" · ");
  inst.stale.style.display = notes.length ? "block" : "none";
  if (reasons.length) {
    // Clear stale OHLC/volume/fan instead of continuing to show a known
    // mismatched chart. Locked entry/TP/SL lines remain visible.
    try {
      inst.candles.setData([]); inst.vol.setData([]);
      inst.fan50.setData([]); inst.fan25.setData([]); inst.fan75.setData([]);
      inst.forecastLabel.textContent = "forecast unavailable · untrusted tape";
      inst.forecastLabel.title = inst.forecastLabel.textContent;
      inst.candles.setMarkers([]);
    } catch {}
    inst.last = null; Own.key = ""; Own.rangeKey = "";
    ownSetLines(inst, d);
    return;
  }
  let bars = tapeArr.filter(b2 => b2.close > 0 && b2.time <= bt0);
  if (bars.length && liveP > 0) {
    // merge the browser's own price prints into the tape: prints in
    // the last bar's bucket UPDATE that bar, later prints open new
    // buckets — the series the chart receives is always unique +
    // ascending and always ends at the live price. (The old bridge
    // APPENDED a second bar at the last bar's own timestamp whenever
    // prints fell inside the current bucket; LWC rejects such a
    // series and the key-guard below then froze the first paint.)
    const lastT = bars[bars.length - 1].time;
    const byTime = new Map(bars.map(b2 => [b2.time, { ...b2 }]));
    for (const pt of (S.priceHist || [])) {
      if (!(pt.p > 0) || pt.t > now) continue;
      const bk = Math.floor(pt.t / bucket0) * bucket0;
      if (bk < lastT) continue;
      const o = byTime.get(bk);
      if (!o) byTime.set(bk, { time: bk, open: pt.p, high: pt.p, low: pt.p, close: pt.p, v: 0 });
      else { o.high = Math.max(o.high, pt.p); o.low = Math.min(o.low, pt.p); o.close = pt.p; }
    }
    bars = [...byTime.values()].sort((x, y) => x.time - y.time);
    const lb = bars[bars.length - 1];
    if (lb.time === bt0) { lb.close = liveP; lb.high = Math.max(lb.high, liveP); lb.low = Math.min(lb.low, liveP); }
    else if (bt0 > lb.time) bars.push({ time: bt0, open: lb.close, high: Math.max(lb.close, liveP),
                     low: Math.min(lb.close, liveP), close: liveP, v: 0 });
  }
  const minVisibleBars = S.focusTf === "tf1" ? 90 : 30;
  if (tapeArr.length < minVisibleBars) {
    // Say WHICH of the two situations this is. "history warming" used to be
    // printed for both, so a genuinely empty tape (backfill never landed)
    // looked like a normal few-second warm-up and the user had no way to
    // tell that no bars were coming. Measured complaints: "history bars are
    // not there" while the banner read "warming".
    const why = tapeArr.length === 0
      ? `no source bars for ${S.symbol} ${S.focusTf.toUpperCase()} — ` +
        `backfill has not delivered any candles (feed: ${S.tfBarsMode || "unknown"}). ` +
        `This is missing data, not a slow load.`
      : `history partial: ${tapeArr.length}/${minVisibleBars} source candles — ` +
        `showing the ${tapeArr.length} that arrived; the rest is still loading`;
    notes.push(why);
    inst.stale.textContent = notes.join(" · ");
    inst.stale.style.display = "block";
  }
  const key = `${S.symbol}|${S.focusTf}|${d.cycle_start}`;
  // A feed handover can replace OHLC history without changing the bar
  // count or last timestamp. Include the server's content/source revision;
  // the old length+time key kept drawing yesterday's price range.
  const sourceRevision = `${S.tfBarsSymbol || S.symbol}|${S.tfBarsRevision?.[S.focusTf] || "unversioned"}`;
  const dataKey = bars.length
    ? `${key}|${sourceRevision}|${bars[bars.length - 1].time}|${bars.length}` : "";
  if (bars.length && dataKey !== Own.key) {
    const cs = bars.map(b2 => ({ time: lt(b2.time), open: b2.open, high: b2.high, low: b2.low, close: b2.close }));
    try {
      inst.candles.setData(cs);
      inst.vol.setData(bars.map(b2 => ({ time: lt(b2.time), value: b2.v || 0,
        color: b2.close >= b2.open ? "rgba(38,166,154,.4)" : "rgba(239,83,80,.4)" })));
      Own.key = dataKey;          // key set ONLY after a clean setData
      const lb = bars[bars.length - 1];
      inst.last = { ...lb };
    } catch { /* rejected series: leave Own.key stale, retry next paint */ }
  }
  // Keep three prior horizon windows plus the current one in view,
  // but never request more history than the server supplied. The 22m
  // chart keeps the documented 90 one-minute-bar first-paint context.
  // Leave room through the forecast horizon so its median path is visible.
  // Keep the same span in view, but express it as a LOGICAL bar range.
  // setVisibleRange(frome/to) depends on both endpoints existing on the
  // time axis; when the server tape started later than the requested
  // window (or the cycle end sat far in the future) the candles were
  // compressed into the left ~15% of the canvas. A logical range pins
  // the data itself to the canvas: `contextBars` of history + the
  // remaining window as future space, independent of timestamps.
  if (bars.length) {
    const N = bars.length;
    const Wsec = Math.max(bucket0, (d.cycle_end || 0) - (d.cycle_start || 0)) || bucket0;
    const minContextBars = (S.focusTf === "tf1" ? 90 : 30);
    const contextBars = Math.max(Math.round(4 * Wsec / bucket0), minContextBars);
    const futureBars = Math.max(2,
      Math.ceil(((d.cycle_end || bt0) - bt0) / bucket0) + 2);
    const sig = `${key}|${N}|${futureBars}`;
    if (sig !== Own.rangeKey || !Own.rangeTo) {
      Own.rangeKey = sig; Own.rangeTo = bt0;
      const from = Math.max(0, N - Math.min(contextBars, N));
      try { inst.chart.timeScale().setVisibleLogicalRange({ from, to: N + futureBars }); } catch {}
      try { inst.chart.timeScale().applyOptions({ rightOffset: 2 }); } catch {}
    }
  }
  // live candle from the tick stream (candles.update, never setData
  // per tick); on a bucket roll the live bar ADVANCES — open = prev
  // close — instead of freezing at the first-painted bar
  const p = liveP;
  if (inst.last && p > 0) {
    const bucket = bucket0;
    const bt = Math.floor(now / bucket) * bucket;
    if (bt > inst.last.time)
      inst.last = { time: bt, open: inst.last.close, high: Math.max(inst.last.close, p),
                    low: Math.min(inst.last.close, p), close: p };
    else {
      const l = inst.last;
      l.close = p; l.high = Math.max(l.high, p); l.low = Math.min(l.low, p);
    }
    const l = inst.last;
    try { inst.candles.update({ time: lt(l.time), open: l.open, high: l.high, low: l.low, close: l.close }); } catch {}
  }
  ownSetLines(inst, d);
  // markers: LOCK at the window boundary + settled hits (server-settled)
  const bucket = (S.focusTf === "tf3" || S.focusTf === "tf4") ? 300 : 60;
  const markers = [];
  if (d.cycle_start) markers.push({ time: lt(Math.floor(d.cycle_start / bucket) * bucket),
    position: (d.direction || 0) < 0 ? "aboveBar" : "belowBar",
    shape: (d.direction || 0) < 0 ? "arrowDown" : "arrowUp",
    color: "#e6b93d", text: "LOCK" });
  for (const t of (d.targets || []))
    if (t.status === "hit" && t.hit_ts) markers.push({ time: lt(Math.floor(t.hit_ts / bucket) * bucket),
      position: "aboveBar", shape: "circle", color: "#22c98a", text: `TP${t.k} hit` });
  const mk = JSON.stringify(markers);
  if (mk !== Own.markerKey) { Own.markerKey = mk;
    try { inst.candles.setMarkers(markers.sort((x, y) => x.time - y.time)); } catch {} }
  // The model's q50 median estimate is always visible. The FAN toggle
  // only adds/removes the q25–q75 context bands; no path is synthesized.
  const b = d.branches, fanOn = !!S.fanOn;
  const fanPts = Array.isArray(b?.fan) ? b.fan : [];
  const fanKey = fanPts.length > 2
    ? `${d.cycle_start}|${Math.round(now)}|${fanOn ? 1 : 0}|${fanPts.length}` : "";
  if (fanKey !== (Own.fanKey || "")) {
    Own.fanKey = fanKey;
    if (!fanKey) {
      inst.fan50.setData([]); inst.fan25.setData([]); inst.fan75.setData([]);
      inst.forecastLabel.textContent = "forecast unavailable · insufficient data";
      inst.forecastLabel.title = inst.forecastLabel.textContent;
    } else {
      const fT = f => Math.floor(now + Number(f) * Math.max(1, (d.cycle_end || now) - now));
      const series = k => {
        const out = [];
        for (const p2 of fanPts) {
          const frac = Number(p2.f), value = Number(p2[k]);
          if (!Number.isFinite(frac) || !Number.isFinite(value) || value <= 0) continue;
          const time = lt(fT(frac));
          if (out.length && time <= out[out.length - 1].time) continue;
          out.push({ time, value });
        }
        return out;
      };
      const median = series("q50");
      let rendered = true;
      try {
        inst.fan50.setData(median.length > 1 ? median : []);
        inst.fan25.setData(fanOn ? series("q25") : []);
        inst.fan75.setData(fanOn ? series("q75") : []);
      } catch {
        rendered = false;
        inst.fan50.setData([]); inst.fan25.setData([]); inst.fan75.setData([]);
      }
      const method = String(b.method || "approximate model");
      const basis = String(b.fan_basis || "");
      const source = [method, basis].filter(Boolean).join(" · ");
      const label = !rendered
        ? "forecast unavailable · chart rejected path"
        : median.length > 1 ? `MODEL ESTIMATE · q50 · ${source}`
          : "forecast unavailable · invalid median path";
      inst.forecastLabel.textContent = label;
      inst.forecastLabel.title = `${label} · not a promise`;
    }
  }
}

/* ═══════ BINANCE chart: the chart you see on binance.com — Binance
   klines (candles + volume + MA7/MA25) from our Binance feed, drawn
   with the vendored lightweight-charts engine. No iframe, no fake. ═══ */
const Bn = { main: null, quads: {}, key: "" };
function bnMake(el, mini) {
  const LW = window.LightweightCharts; if (!LW || !el) return null;
  const chart = LW.createChart(el, {
    layout: { background: { color: "transparent" }, textColor: "#cbd5e1", fontSize: 9 },
    grid: { vertLines: { color: "rgba(140,140,170,.08)" }, horzLines: { color: "rgba(140,140,170,.08)" } },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: "rgba(140,140,170,.25)" },
    width: el.clientWidth || 300, height: el.clientHeight || (mini ? 142 : 340),
  });
  // user: Binance LINE chart, not candles — closes as a line
  const levelValues = [];
  const candles = chart.addLineSeries({ color: "#f0b90b", lineWidth: 2,
    autoscaleInfoProvider: levelAutoscaleProvider(levelValues),
    priceLineVisible: true, lastValueVisible: true });
  const vol = chart.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "volume" } });
  chart.priceScale("").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
  const ma = (color) => chart.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
  const out = { chart, candles, vol, ma7: ma("rgba(240,185,11,.85)"), ma25: ma("rgba(200,120,255,.7)"), levelValues, _lines: [], last: null };
  new ResizeObserver(() => { try { chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }); } catch {} }).observe(el);
  return out;
}
function bnFeed(inst, bars) {
  if (!inst || !bars.length) return;
  const cs = bars.map(b => ({ time: lt(b.t), open: b.o, high: b.h, low: b.l, close: b.c }));
  inst.candles.setData(cs.map(c => ({ time: c.time, value: c.close })));
  inst.vol.setData(bars.map(b => ({ time: lt(b.t), value: b.v || 0,
    color: b.c >= b.o ? "rgba(38,166,154,.4)" : "rgba(239,83,80,.4)" })));
  const maD = (n) => cs.map((c, i) => i < n - 1 ? null
    : { time: c.time, value: cs.slice(i - n + 1, i + 1).reduce((x, y) => x + y.close, 0) / n }).filter(Boolean);  // cs times already lt()-shifted
  inst.ma7.setData(maD(7)); inst.ma25.setData(maD(25));
  const lb = bars[bars.length - 1];
  // lb.t is the raw epoch; the series is drawn with lt(lb.t). Storing the
  // raw value made every live update call lt() a SECOND time, shifting
  // the tick an hour into the future and freezing the chart.
  inst.last = { time: lt(lb.t), open: lb.o, high: lb.h, low: lb.l,
                close: lb.c, v: lb.v || 0, bucket: Number(lb.t) };
  try { inst.chart.timeScale().scrollToRealTime(); } catch {}
}
async function bnLoad(tf, inst, el) {
  try {
    const d = await api(`/api/candles?symbol=${S.symbol}&tf=${tf}&n=180`);
    if (S.chartSrc === "binance" && d.bars) bnFeed(inst, d.bars);
  } catch { /* feed warms up */ }
}
function bnEnsure() {
  if (S.chartSrc !== "binance") return;
  const mainEl = $("#chart-binance");
  if (!mainEl) return;
  // The TF cards no longer host a Binance copy (it was created inside a
  // hidden slot and stayed empty) — only the big chart is built here.
  if (!Bn.main) {
    if (mainEl.classList.contains("hidden")) return;   // build when visible
    Bn.main = bnMake(mainEl, false);
    try { Bn.main.chart.applyOptions({ width: mainEl.clientWidth || 800,
                                       height: mainEl.clientHeight || 340 }); } catch (e) {}
  }
  if (Bn.main) bnLoad(S.focusTf, Bn.main, mainEl);
}
function bnTickMain() {
  const inst = Bn.main, p = S.prices?.[S.symbol];
  if (inst && !inst.last && Date.now() - (Bn._tryAt || 0) > 5000) {
    Bn._tryAt = Date.now(); bnLoad(S.focusTf, inst);
  }
  if (!inst || !inst.last || !(p > 0)) return;
  const l = inst.last;
  // bucket rollover: the Binance series is 1-minute, so a new minute must
  // append a bar (previously the last bar was rewritten forever).
  const nb = Math.floor(serverNow());
  if (inst.bucket && nb > inst.bucket) {
    bnLoad(S.focusTf, inst);
    return;
  }
  l.close = p; l.high = Math.max(l.high, p); l.low = Math.min(l.low, p);
  try { inst.candles.update({ time: l.time, value: l.close }); }
  catch { bnLoad(S.focusTf, inst); }   // chart rejected the tick: refetch
}

/* position sizing from the REAL locked stop distance per TF */
function renderRiskCalc() {
  const out = $("#risk-out"), tbl = $("#risk-table");
  if (!out) return;
  try {
    const eqSaved = localStorage.getItem("riskEq"), pctSaved = localStorage.getItem("riskPct");
    if (eqSaved && document.activeElement !== $("#risk-eq")) $("#risk-eq").value = eqSaved;
    if (pctSaved && document.activeElement !== $("#risk-pct")) $("#risk-pct").value = pctSaved;
  } catch {}
  const eq = Number($("#risk-eq")?.value || 0), pct = Number($("#risk-pct")?.value || 0);
  const a = S.snap?.assets?.[S.symbol];
  if (!a || !(eq > 0) || !(pct > 0)) { out.textContent = "—"; return; }
  const riskUsdt = eq * pct / 100;
  const row = tf => {
    const c = a.tfs?.[tf];
    if (!c?.entry || !c?.sl) return null;
    const dist = Math.abs(c.entry - c.sl);
    if (!(dist > 0)) return null;
    const qty = riskUsdt / dist, notional = qty * c.entry;
    return { tf, qty, notional, lev: notional / eq };
  };
  const f = row(S.focusTf);
  out.textContent = f ? `${f.qty.toFixed(4)} ${S.symbol} · ${fmt(f.notional, 0)} USDT notional · ${f.lev.toFixed(2)}× equity`
                      : "no locked stop on this TF yet";
  const tfEl = $("#risk-out-tf"); if (tfEl) tfEl.textContent = `size for ${S.focusTf.toUpperCase()} (risk ${fmt(riskUsdt, 2)} USDT)`;
  if (tbl) tbl.innerHTML = TF.map(tf => {
    const r = row(tf);
    return r ? `${tf.toUpperCase()}: ${r.qty.toFixed(4)} (${fmt(r.notional, 0)} USDT)` : `${tf.toUpperCase()}: —`;
  }).join(" · ");
}
async function applyRiskToTrading() {
  const eq = Number($("#risk-eq")?.value || 0), pct = Number($("#risk-pct")?.value || 0);
  const st = $("#risk-apply-state");
  if (!(eq > 0) || !(pct > 0)) { if (st) st.textContent = "enter equity and risk % first"; return; }
  const r = await api("/api/trading/config", { method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sizing_mode: "risk", equity_usdt: eq, risk_pct: pct }) });
  if (st) st.textContent = r.ok
    ? `ACTIVE: orders sized as ${fmt(eq,0)} USDT x ${pct}% / stop distance (capped by max order + equity). Binance must still be enabled + ARMED.`
    : "could not save sizing — " + (r.error || "");
  toast(r.ok ? "position sizing is now LIVE for Binance orders" : "sizing save failed", r.ok ? "ok" : "err");
}
function bindRiskCalc() {
  const ab = $("#risk-apply");
  if (ab && !ab.dataset.wired) { ab.dataset.wired = "1"; ab.addEventListener("click", applyRiskToTrading); }
  for (const id of ["#risk-eq", "#risk-pct"]) {
    const el = $(id); if (!el || el.dataset.wired) continue;
    el.dataset.wired = "1";
    el.addEventListener("input", () => {
      try {
        localStorage.setItem("riskEq", $("#risk-eq").value);
        localStorage.setItem("riskPct", $("#risk-pct").value);
      } catch {}
      renderRiskCalc();
    });
  }
}

function renderBranches() {
  const stats = $("#branch-stats"), cvHost = $("#branch-canvas");
  if (!stats || !cvHost) return;
  const a = S.snap?.assets?.[S.symbol];
  const d = a?.tfs?.[S.focusTf];
  const b = d?.branches;
  $("#branch-scope").textContent = `${S.symbol} · ${S.focusTf.toUpperCase()} · `
    + (b?.n ? `measured first-touch · n=${b.n}` : b?.method ? String(b.method) : "collecting");
  if (!b || !a) {
    cvHost._pc = null;
    stats.innerHTML = "";
    cvHost.innerHTML = `<div class="dim pad" style="font-size:11px">waiting for a direction + TP/SL on ${S.focusTf.toUpperCase()}…</div>`;
    $("#branch-note").textContent =
      "bands appear once the window has a direction and a seeded stop";
    return;
  }
  const _pct = p => !Number.isFinite(p) ? "—" : p > 0 && p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`;
  // Round a single first-touch distribution consistently across cards and bar.
  const probs = [b.p_target, b.p_between, b.p_stop];
  let percentages = null;
  if (probs.every(p => Number.isFinite(p) && p >= 0)) {
    const total = probs.reduce((a, v) => a + v, 0);
    if (total > 0) {
      const raw = probs.map(p => p * 100 / total), rounded = raw.map(Math.floor);
      const order = raw.map((v, i) => [v - rounded[i], i]).sort((a, b) => b[0] - a[0]);
      for (let n = 100 - rounded.reduce((a, v) => a + v, 0), i = 0; i < n; i++) rounded[order[i][1]]++;
      percentages = rounded;
    }
  }
  const outcomePct = i => percentages ? `${percentages[i]}%` : "—";
  stats.innerHTML = `
    <div class="kv"><span>P(target before stop)</span><b class="acc-up">${outcomePct(0)}${b.cal_delta ? ` <span class="dim" style="font-size:11px">calibrated separately ${_pct(b.p_target_cal)}</span>` : ""}</b></div>
    <div class="kv"><span>calibration (graded forecasts)</span><b>${S.snap?.prob_audit?.n ? `Δ ${S.snap.prob_audit.calibration_delta > 0 ? "+" : ""}${Math.round((S.snap.prob_audit.calibration_delta || 0) * 100)} pts · Brier raw ${S.snap.prob_audit.brier ?? "—"}${S.snap.prob_audit.brier_cal != null ? ` vs cal ${S.snap.prob_audit.brier_cal}` : ""} · n=${S.snap.prob_audit.n}` : "collecting — active after 30 graded windows"}</b></div>
    <div class="kv"><span>P(stop first)</span><b class="acc-dn">${outcomePct(2)}</b></div>
    <div class="kv"><span>P(target | similar setups)</span><b class="acc-up">${b.p_target_cond != null ? `${_pct(b.p_target_cond)} <span class="dim" style="font-size:11px">n=${b.n_cond} · EV ${b.ev_bps_cond > 0 ? "+" : ""}${fmt(b.ev_bps_cond, 1)} bp</span>` : `<span class="dim">collecting (n=${b.n_cond || 0}/30 matched)</span>`}</b></div>
    <div class="kv"><span>fan basis</span><b>${esc(b.fan_basis || "all windows")}${b.drift_bps != null ? ` · q50 ${b.drift_bps >= 0 ? "+" : ""}${fmt(b.drift_bps, 1)} bp <span class="${b.fan_drift === "against" ? "acc-dn" : b.fan_drift === "with" ? "acc-up" : "dim"}">(${esc(b.fan_drift || "—")})</span>` : ""}</b></div>
    <div class="kv"><span>P(neither barrier touched)</span><b>${outcomePct(1)}</b></div>
    <div class="kv"><span>P(close down)</span><b>${Math.round(b.p_close_down * 100)}%</b></div>
    <div class="kv"><span>σ (remaining)</span><b>${px(b.sigma)}</b></div>
    <div class="kv"><span>expectancy (barrier pair)</span><b class="${(b.ev_bps ?? 0) >= 0 ? "acc-up" : "acc-dn"}">${b.ev_bps != null ? `${b.ev_bps >= 0 ? "+" : ""}${fmt(b.ev_bps, 1)} bp` : "—"}</b></div>
    <div class="kv"><span>P(target) 95% interval</span><b>${b.ci_target ? `${Math.round(b.ci_target[0] * 100)}–${Math.round(b.ci_target[1] * 100)}%` : "—"}</b></div>
    <div class="kv"><span>forecast audit (all windows)</span><b>${S.snap?.prob_audit?.n ? `n=${S.snap.prob_audit.n} · pred ${Math.round(S.snap.prob_audit.mean_predicted * 100)}% vs realised ${Math.round(S.snap.prob_audit.realised_tp_first * 100)}% · ${esc(S.snap.prob_audit.verdict || "")}` : "grading after settled windows"}</b></div>
    <div class="kv"><span>basis</span><b>${b.method === "empirical-first-touch" ? `measured · n=${b.n} windows` : "approx · thin tape"}</b></div>
    <div class="kv"><span>time left</span><b>${Math.round(b.rem_s / 60)}m ${Math.round(b.rem_s % 60)}s</b></div>`;
  $("#branch-note").textContent = b.n
    ? `measured: first-touch outcomes replayed over ${b.n} historical windows on this asset's own tape (both-barrier bars count as stop-first, ${b.ambig} ambiguous) · bands = empirical return quantiles (${b.fan_basis || "all windows"}) · realised path in white${b.resolved ? ` · ${b.resolved.barrier} → window decided` : ""}`
    : "thin tape — driftless approximation until 20+ windows can be replayed · realised path in white";
  // ---- probability bar: the answer in one glance ----
  const bar = $("#branch-bar");
  if (bar) {
    // ONE measured triple per bar (audit §4.1): the same largest-remainder
    // rounding the cards use, so bar and cards always agree and sum to
    // exactly 100. (The old bar mixed CONDITIONAL P(target) with
    // UNCONDITIONAL P(stop) and printed 11 + 76 + 19 = 106 %.)
    if (!percentages) { bar.innerHTML = ""; return; }
    const [pt, pb, ps] = percentages;
    const seg = (v, color, label) => v > 0 ? `<div style="flex:${v};background:${color};color:#0b0b14;font:700 10px ui-monospace,monospace;text-align:center;padding:4px 0;white-space:nowrap;overflow:hidden">${label} ${v}%</div>` : "";
    bar.innerHTML = `<div style="display:flex;border-radius:6px;overflow:hidden;gap:1px">
      ${seg(pt, "#26a69a", "TARGET FIRST")}${seg(pb, "#3a3f55", "NEITHER")}${seg(ps, "#ef5350", "STOP FIRST")}</div>
      <div class="hint" style="margin-top:3px">of ${b.n} past windows on this asset with these TP/SL distances — green hit target first, red hit stop first, grey neither barrier was touched</div>`;
  }
  // ---- probability chart: history + measured fan, drawn by the
  // shared PriceChart engine (static/LineChart.js) ----
  const t0 = d.cycle_start || (Date.now() / 1000 - 60);
  const t1 = d.cycle_end || (Date.now() / 1000 + b.rem_s);
  const tN = b.now || (Date.now() / 1000);
  const hist = realisedSeries(t0 - 1, Date.now() / 1000);
  const lastP = hist.length ? hist[hist.length - 1].p : (b.entry || a.price || 0);
  // ── plain-English takeaway: the panel must answer "so what?" ──
  const fanEnd = (b.fan && b.fan.length) ? b.fan[b.fan.length - 1] : null;
  const pT = Math.round((b.p_target_cond ?? b.p_target ?? 0) * 100);
  const pS = Math.round((b.p_stop ?? 0) * 100);
  const medTxt = fanEnd && Number.isFinite(fanEnd.q50)
    ? `most likely price at the ${Math.round((b.rem_s || 0) / 60) || TF_META[S.focusTf].mins}-minute mark is <b>${px(fanEnd.q50)}</b>` : "";
  const rangeTxt = fanEnd && Number.isFinite(fanEnd.q25) && Number.isFinite(fanEnd.q75)
    ? `the middle half of outcomes lands between <b>${px(fanEnd.q25)}</b> and <b>${px(fanEnd.q75)}</b>` : "";
  const tk = $("#branch-takeaway");
  if (tk) {
    const verdict = pT >= 60 ? `favours the TARGET (${pT}% target first vs ${pS}% stop first)`
      : pS >= 60 ? `favours the STOP (${pS}% stop first vs ${pT}% target first)`
        : `is a coin-flip (${pT}% target first vs ${pS}% stop first)`;
    const resolvedTxt = b.resolved
      ? `<b class="acc-warn">${esc(b.resolved.barrier || "barrier observed")}</b> — this window is already decided, so P(target first) is a fact (${pT}%), not a forecast. `
      : "";
    tk.innerHTML = `<b>IN PLAIN ENGLISH</b> — ${resolvedTxt}from ${b.entry ? px(b.entry) : "entry"} with TP ${b.tp ? px(b.tp) : "—"} and SL ${b.sl ? px(b.sl) : "—"}, `
      + `the measured path ${verdict}. ${medTxt ? medTxt + "; " : ""}${rangeTxt ? rangeTxt + " — the shaded bands below show the full 10th–90th percentile spread. " : ""}`
      + `<span class="dim">${b.method === "empirical-first-touch" ? `Measured on ${b.n} replayed windows of this asset's own tape, not a model guess.` : b.n ? `Measured on ${b.n} windows (${esc(b.fan_basis || "all windows")}).` : "Thin tape — this is a driftless approximation until 20+ windows can be replayed."}</span>`;
  }
  // ── zones: green TARGET band (price → TP), red STOP band (price → SL) ──
  const zones = [];
  if (b.entry && b.tp) zones.push({ from: b.entry, to: b.tp, label: "TARGET ZONE",
    color: "rgba(38,166,154,.10)", text: "rgba(120,230,190,.85)" });
  if (b.entry && b.sl) zones.push({ from: b.entry, to: b.sl, label: "STOP ZONE",
    color: "rgba(240,56,90,.10)", text: "rgba(255,140,160,.85)" });
  const lg = $("#branch-legend");
  if (lg) {
    lg.innerHTML = `<span class="lg"><i style="background:#f2f2fa"></i>realised price</span>`
      + `<span class="lg"><i style="background:#7fd8cf"></i>median (50th)</span>`
      + `<span class="lg"><i style="background:rgba(95,200,160,.6)"></i>bullish band 75th–90th</span>`
      + `<span class="lg"><i style="background:rgba(240,90,110,.6)"></i>bearish band 10th–25th</span>`
      + `<span class="lg"><i style="background:#26a69a"></i>TP</span>`
      + `<span class="lg"><i style="background:#f0385a"></i>SL</span>`
      + `<span class="lg"><i style="background:#c9a227"></i>NOW</span>`;
  }
  if (!cvHost._pc) { cvHost.innerHTML = ""; cvHost._pc = new PriceChart(cvHost); }
  cvHost._pc.setModel({
    series: [{ pts: hist, color: "#f2f2fa" }],
    zones,
    levels: [
      ...(b.tp ? [{ p: b.tp, label: "TP", color: "#26a69a", dash: true }] : []),
      ...(b.entry ? [{ p: b.entry, label: "ENTRY", color: "#e6b93d" }] : []),
      ...(b.sl ? [{ p: b.sl, label: "SL", color: "#f0385a", dash: true }] : []),
    ],
    fan: (b.fan && b.fan.length > 2 && hist.length)
      ? { originT: tN, originP: lastP,
          pts: b.fan.map(p2 => ({ t: tN + p2.f * Math.max(1, t1 - tN),
                                  lo: p2.q05, lo2: p2.q25, mid: p2.q50, hi2: p2.q75, hi: p2.q95 })) }
      : null,
    fanLabels: true,
    nowT: tN, t0, t1, ma: 0, domainFan: "inner",
    ariaLabel: `${S.symbol} ${S.focusTf} probability chart: realised price ` +
      `with measured outcome fan to cycle end`,
    emptyText: "collecting realised path…",
  });
}

/* ═══════ formula live-value ticker (raw → new) ═══════ */
function tickFormulaValues() {
  const fv = S.snap?.formula_values?.[`${S.symbol}|${S.focusTf}`];
  const cur = fv?.raw || {};
  const prev = S.prevFV;
  S.prevFV = { ...cur };
  // keep the LIVE sample rings fed (15 s snapshots + analyse runs) so the
  // formula charts keep moving between GRCs
  try {
    if (Object.keys(cur).length) {
      if (!S.fvRings || S.fvRingKey !== `${S.symbol}|${S.focusTf}`) {
        S.fvRingKey = `${S.symbol}|${S.focusTf}`;
        S.fvRings = {};
        for (const k of Object.keys(S.fvBackfill || {})) {
          S.fvRings[k] = (S.fvBackfill[k] || []).slice(-100);
        }
      }
      for (const k of Object.keys(cur)) {
        if (!S.fvRings[k]) S.fvRings[k] = [];
        S.fvRings[k].push(cur[k]);
        if (S.fvRings[k].length > 140) S.fvRings[k] = S.fvRings[k].slice(-140);
      }
    }
  } catch (e) { /* rings optional */ }
  const box = $("#fl-ticker");
  if (!box || !prev) return;
  const names = fv?.names || {};
  const lines = [];
  const nowT = new Date().toLocaleTimeString();
  for (const k of Object.keys(cur)) {
    const a2 = prev[k], b2 = cur[k];
    if (a2 == null || b2 == null) continue;
    if (Math.abs(a2 - b2) < 1e-9) continue;
    const up = b2 > a2;
    lines.push(`${nowT} F${k} ${names[k] || ""} ${fx(a2, 4)} → ${up ? "▲" : "▼"} ${fx(b2, 4)}`);
    if (lines.length >= 6) break;
  }
  if (!lines.length) return;
  box.innerHTML = lines.map(l =>
    `<div style="color:${l.includes("▲") ? "var(--green)" : "var(--red)"}">${esc(l)}</div>`
  ).join("") + box.innerHTML;
  // cap the log at ~40 lines
  const kids = box.children;
  for (let i = 40; i < kids.length; i++) kids[i].remove();
}

/* ═══════ tab explainer (qwen2.5-coder local / deterministic) ═══════ */
function ensureExplainPanel(v) {
  const sec = $("#view-" + v);
  if (!sec) return null;
  let p = $("#ex-panel-" + v);
  if (!p) {
    p = document.createElement("div");
    p.className = "panel"; p.id = "ex-panel-" + v;
    p.innerHTML = `<div class="panel-h">WHY — PLAIN-LANGUAGE READOUT
        <span class="dim" id="ex-src-${v}"></span>
        <button class="btn" id="ex-btn-${v}" style="float:right;padding:2px 10px">re-explain</button></div>
      <div id="ex-${v}" class="logic" style="white-space:pre-wrap">loading…</div>`;
    const head = sec.querySelector(".tab-head");
    if (head && head.nextSibling) sec.insertBefore(p, head.nextSibling);
    else sec.prepend(p);
    const b = $("#ex-btn-" + v);
    if (b) b.addEventListener("click", () => loadExplain(v, true));
  }
  return p;
}
async function loadExplain(view, force) {
  const box = view === "dashboard" ? $("#why-explain") : $("#ex-" + view);
  if (!box) return;
  try {
    const out = await api("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ view, symbol: S.symbol, tf: S.focusTf }),
    });
    box.textContent = out.text || "(no explanation)";
    const src = $("#ex-src-" + view);
    if (src) src.textContent = (out.cached ? "cached · " : "") +
      (out.source === "qwen2.5-coder" ? "🧠 qwen2.5-coder:3b (local)" :
       "local fallback · model: " + (out.model_status || "starting…"));
    return out;
  } catch (e) {
    box.textContent = "explain failed: " + e.message;
  }
}
window.toggleWhyExplain = function () {
  const pre = $("#why-explain");
  if (!pre) return;
  const show = pre.style.display === "none";
  pre.style.display = show ? "block" : "none";
  if (show && !pre.textContent) loadExplain("dashboard");
};
function syncAnalyseButton() {
  // dashboard-only chrome (user request): the ANALYSE button AND the
  // ASSET / TIMEFRAME pickers — other tabs follow the dashboard's
  // selection and show their scope in the tab title (BTC · TF1 · 22M)
  const onDash = S.view === "dashboard";
  const b = $("#btn-analyse");
  if (b) b.style.display = onDash ? "" : "none";
  const sa = $("#seg-assets"), st = $("#seg-tf");
  if (sa) sa.style.display = onDash ? "" : "none";
  if (st) st.style.display = onDash ? "" : "none";
}
function setView(v) {
  S.view = v; S.lastViewFetch = 0;
  $$(".tab-btn").forEach(x => x.classList.toggle("active", x.dataset.view === v));
  $$(".view").forEach(x => x.classList.toggle("active", x.id === "view-" + v));
  syncAnalyseButton();
  refreshActiveView(true);
}
function buildTabs() {
  $$(".tab-btn").forEach(b => b.addEventListener("click", () => {
    S.view = b.dataset.view;
    $$(".tab-btn").forEach(x => x.classList.toggle("active", x === b));
    $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + S.view));
    syncAnalyseButton();
    refreshActiveView(true);
  }));
}
function buildToolbar() {
  const sa = $("#seg-assets"), st = $("#seg-tf");
  if (!sa || !st) return;
  ASSETS.forEach(sym => {
    const b = document.createElement("button");
    b.className = "seg-btn" + (sym === S.symbol ? " active" : "");
    b.dataset.sym = sym; b.textContent = sym;
    b.addEventListener("click", () => setSymbol(sym));
    sa.appendChild(b);
  });
  TF.forEach(tf => {
    const m = TF_META[tf];
    const b = document.createElement("button");
    b.className = "seg-btn" + (tf === S.focusTf ? " active" : "");
    b.dataset.tf = tf;
    b.innerHTML = `${tf.toUpperCase()}<small>${m.mins}m</small>`;
    b.addEventListener("click", () => setFocusTf(tf));
    st.appendChild(b);
  });
}
function syncToolbar() {
  $$("#seg-assets .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.sym === S.symbol));
  $$("#seg-tf .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.tf === S.focusTf));
}
function setFocusTf(tf) {
  if (!TF.includes(tf)) return;
  S.focusTf = tf;
  try { localStorage.setItem("focusTf", tf); } catch (e) { /* private */ }
  // choosing a timeframe switches the DASHBOARD to focus mode:
  // that window + its sub-timeframe TPs only (⊞ all timeframes reverts)
  S.tfFocus = true;
  try { localStorage.setItem("tfFocus", "1"); } catch (e) { /* private mode */ }
  syncToolbar(); renderWhy(); renderQuads(); S.lastViewFetch = 0;
  // repaint the hero IMMEDIATELY for the new TF: the previous window's
  // TP/SL must never survive the click while waiting for the next snapshot
  try { renderHero(); } catch (e) { console.error("hero", e); }
  try { applyChartSrc(); } catch (e) { /* main TV widget follows the focused TF */ }
  loadCandles().catch(() => {});
  refreshActiveView(true);
}

function renderDashboardMode() {
  // REWRITTEN dashboard: the decision hero (focused-TF big chart) is
  // ALWAYS visible and the four timeframe cards sit under it as a
  // switcher strip — the old mode hid one or the other, so the page
  // looked unchanged between modes and the big chart was easy to miss.
  const hero = $("#tf-hero"), grid = $("#quad-grid");
  if (!hero || !grid) return;
  hero.classList.remove("hidden");
  grid.style.display = "";
  renderHero();
}

function _pWinRow(res) {
  // Honest confidence: the isotonic map fitted on settled ledger rows.
  // Until it has enough samples the row says so instead of printing the
  // raw score twice under a different name.
  const pc = res?.p_correct;
  const cs = res?.calib_iso || {};
  const cal = pc != null
    ? `<b>${fx(pc, 3)} <span class="dim" style="font-size:11px">P(win) · ${esc(cs.source || "map")} n=${cs.n ?? 0}${cs.slope != null ? ` · slope ${fx(cs.slope, 2)}` : ""}</span></b>`
    : `<b class="dim">needs 40 settled · ${esc(cs.source || "insufficient")} n=${cs.n ?? 0}</b>`;
  const sp = res?.spillover || {};
  const spTxt = sp.available
    ? `<b class="${Number(sp.score) > 0 ? "acc-up" : Number(sp.score) < 0 ? "acc-dn" : "dim"}">${fx(sp.score, 2)}
        <span class="dim" style="font-size:11px">spillover · ${Math.round((sp.coverage || 0) * 100)}% inputs
        ${sp.xasset ? `· BTC6h ${(sp.xasset.ret * 100).toFixed(2)}%` : ""}</span></b>`
    : `<b class="dim">spillover — (thin tape)</b>`;
  const hz = res?.horizon || {};
  const hzRow = hz.state
    ? `<div class="kv"><span>horizon</span><b class="${hz.state === "tradable" ? "" : hz.state === "marginal" ? "acc-warn" : "acc-dn"}">${esc(hz.state)}
        <span class="dim" style="font-size:11px">σ ${hz.sigma_bps ?? "—"}bp vs ${hz.required_bps ?? "—"}bp needed · ${esc(hz.note || "")}</span></b></div>`
    : "";
  const eg = res?.edge || {};
  // cost gate: the ONLY tradeability claim the UI makes. Gross expectancy
  // is read from the out-of-fold registry (never the live fit), so a lane
  // that cannot pay its own fees says so instead of printing a nice score.
  const edgeRow = eg.state
    ? `<div class="kv"><span>cost gate</span><b class="${eg.state === "open" ? "" : eg.state === "unmeasured" ? "dim" : eg.state === "thin" ? "acc-warn" : "acc-dn"}">${esc(eg.state)}
        <span class="dim" style="font-size:11px">${eg.state === "unmeasured"
          ? `${esc(eg.note || "no measurement yet")}`
          : `OOF gross ${fx(eg.gross_bp, 1)}bp vs ${fx(eg.friction_bps, 1)}bp cost · net ${fx(eg.expected_net_bp, 1)}bp${eg.state === "blocked" ? ` · needs ${fx(eg.required_gross_bp, 1)}bp` : ""} · ${esc(eg.state === "blocked" ? "fee not covered" : eg.source || "")}`}</span></b></div>`
    : "";
  const sv = (res?.sleeves || []).filter(x => x && x.state !== "blocked");
  // event sleeves carry their own verdict; "unproven" is shown as such
  const sleeveRow = sv.length
    ? sv.slice(0, 2).map(x => `<div class="kv"><span>sleeve · ${esc(x.name || "")}</span><b class="${x.state === "open" ? "" : "acc-warn"}">${esc(x.state)}
        <span class="dim" style="font-size:11px">${fx(x.gross_bp, 1)}bp gross vs ${fx(x.fee_bp, 1)}bp cost · net ${fx(x.expected_net_bp, 1)}bp · PF ${fx(x.profit_factor, 2)} · n ${x.n ?? "—"} · boot t ${fx(x.boot_t, 2)} · folds ${esc(x.folds || "—")}</span></b></div>`).join("")
    : "";
  const mi = res?.micro || {};
  const micRow = (mi && Object.keys(mi).length)
    ? `<div class="kv"><span>microstructure</span><b class="dim" style="font-size:11px">OFI ${fx(mi.ofi, 2)} · persist ${fx(mi.ofi_persist, 2)} · VPIN ${fx(mi.vpin, 2)} · size skew ${fx(mi.size_skew, 2)} · Hawkes ${fx(mi.hawkes, 2)} · friction ${fx(mi.friction_bps, 1)}bp</b></div>`
    : "";
  const vb = res?.vol_barriers || {};
  const vbRow = (vb.tp_bps != null)
    ? `<div class="kv"><span>vol barriers</span><b class="${vb.ok ? "" : "acc-warn"}">TP ${fx(vb.tp_bps, 1)}bp · SL ${fx(vb.sl_bps, 1)}bp
        <span class="dim" style="font-size:11px">${esc(vb.reason || "")}</span></b></div>` : "";
  return `<div class="kv"><span>P(win)</span>${cal}</div>
    <div class="kv"><span>cross-asset</span>${spTxt}</div>${hzRow}${edgeRow}${sleeveRow}${micRow}${vbRow}`;
}

function renderBook() {
  const block = S.snap?.sleeves || {};
  const book = block.portfolio;
  const wrap = $("#book-sleeves");
  if (!wrap) return;
  const modeEl = $("#book-mode");
  const meta = $("#book-meta");
  const stats = $("#book-stats");

  if (!book || book.error) {
    if (modeEl) modeEl.textContent = "";
    if (meta) meta.textContent = "";
    if (stats) stats.innerHTML = "";
    wrap.innerHTML = `<div class="dim pad">${book?.error ? "paper book failed: " + book.error : "paper book warming up…"}</div>`;
    const p0 = $("#book-positions"); if (p0) p0.innerHTML = "";
    const a0 = $("#book-assets"); if (a0) a0.innerHTML = "";
    return;
  }

  // Bug fixed 2026-11-26: these three formatters disagreed with each other
  // and with the global fx(). `bp(NaN)` printed the literal string "NaN bp"
  // into the panel, `pct(NaN)` printed "0.00%" - which reads as a REAL
  // measured zero rather than missing data - and `pct(null)` invented 0.
  // A number that failed to compute must render as "—" (unknown), never as
  // a zero and never as the word NaN. All three now share one rule.
  const _unknown = v => v == null || !isFinite(Number(v));
  const pct = v => (_unknown(v) ? "—" : `${(Number(v) * 100).toFixed(2)}%`);
  const bp = v => (_unknown(v) ? "—"
    : `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(1)} bp`);
  const cls = v => (_unknown(v) ? "dim"
    : Number(v) > 0 ? "acc-up" : Number(v) < 0 ? "acc-dn" : "dim");

  // ---- observe vs enforced: the single most important honesty label ----
  const enforced = book.monitors_enforced !== false;
  if (modeEl) {
    // A PAUSE MUST NAME ITS CAUSE (user report 2026-11-27: "why paper
    // execution is paused?"). The badge used to be the whole answer. It now
    // prints the contract when fills are live, and the specific per-asset
    // fact when they are not.
    const _paused = book.execution_status === "paused";
    const _reasons = Object.entries(book.paused_assets || {});
    modeEl.innerHTML = _paused
      ? `<b class="acc-warn">PAPER EXECUTION PAUSED</b>`
      : `<b class="acc-up">PAPER EXECUTION LIVE</b> · ${enforced ? "enforced" : "observe"}`;
    modeEl.title = _paused
      ? (book.execution_reason || "Paper execution is paused")
      : (book.execution_reason || "verified next-bar-open fills");
    const _why = $("#book-pause-why");
    if (_why) {
      _why.innerHTML = _paused
        ? `<b class="acc-warn">why no fills:</b> ` +
          (_reasons.length
            ? _reasons.map(([a, w]) => `<span class="mono">${esc(a)}</span> — ${esc(w)}`).join(" · ")
            : esc(book.execution_reason || "unknown reason — this is itself a defect"))
        : `<b class="acc-up">fill contract:</b> ${esc(Object.entries(book.execution_contract || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "not published")}`;
    }
  }
  if (meta) {
    meta.textContent = (book.execution_status === "paused" ? `${book.execution_reason} ` : "")
      + `· gross ${pct(book.gross_exposure)}`
      + ` / max ${pct(book.limits?.max_gross || 1)}`
      + ` · ${book.closed_trades || 0} recorded fills`
      + (book.avg_slippage_bps != null ? ` · slip ${Number(book.avg_slippage_bps).toFixed(1)} bp` : "");
  }

  // ---- headline stats ----
  if (stats) {
    const unreal = (book.open_positions || [])
      .reduce((a, p) => a + Number(p.unrealised_bps || 0) * Number(p.size || 0), 0);
    stats.innerHTML = [
      ["equity", `${(Number(book.equity || 1) * 100).toFixed(2)}%`, cls(Number(book.equity || 1) - 1)],
      ["realized", bp(book.realized_bps), cls(book.realized_bps)],
      ["open p&l", bp(unreal), cls(unreal)],
      ["drawdown", pct(book.drawdown), Number(book.drawdown) > 0.05 ? "acc-dn" : "dim"],
      ["positions", String((book.open_positions || []).length), "dim"],
    ].map(([k, v, c]) => `<div class="kv"><span>${k}</span><b class="${c}">${v}</b></div>`).join("");
  }

  // ---- sleeper rows: live edge next to the certified number ----
  const rows = (book.sleeves || []).map(r => {
    const st = String(r.status || "open").toUpperCase();
    const c = r.status === "open" ? "acc-up" : r.status === "halted" ? "acc-dn" : "acc-warn";
    const live = r.live_trades
      ? `${bp(r.live_ev_bps)} · t ${num(r.live_t)} · n=${r.live_trades} · win ${r.live_win_rate == null ? "—" : `${(Number(r.live_win_rate) * 100).toFixed(0)}%`}`
      : "no live fills yet";
    const why = (r.halt_reasons || [])[0];
    return `<div class="kv"><span>${r.sleeve_id}</span>`
      + `<b class="${c}">${st}</b>`
      + `<span class="dim">certified ${bp(r.research_ev_bps)} · PF ${num(r.research_pf)}</span>`
      + `<span class="dim">${live}</span>`
      + (why ? `<span class="dim acc-warn" style="text-transform:none">${why}</span>` : "")
      + `</div>`;
  }).join("");
  wrap.innerHTML = rows || `<div class="dim pad">no certified sleeves in the book</div>`;

  // ---- open positions: entry, invalidation stop, hold timer ----
  const posWrap = $("#book-positions");
  if (posWrap) {
    const open = (book.open_positions || []).map(p => {
      const total = Math.max(1, Number(p.hold_bars || 1));
      const left = Math.max(0, Number(p.bars_remaining ?? total));
      const done = Math.min(100, 100 * (total - left) / total);
      const tag = Number(p.unrealised_bps) >= 0 ? "acc-up" : "acc-dn";
      return `<div class="kv"><span>${p.sleeve_id} · ${p.asset}</span>`
        + `<b class="${p.direction > 0 ? "acc-up" : "acc-dn"}">${p.direction > 0 ? "LONG" : "SHORT"}`
        + ` <span class="${tag}">${bp(p.unrealised_bps)}</span></b>`
        + `<span class="dim">entry ${px(p.entry_price)} · stop ${px(p.stop)}</span>`
        + `<span class="dim">${(Number(p.size) * 100).toFixed(1)}% · edge ${bp(p.expected_edge_bps)}</span>`
        + `<div class="bar" title="${left} of ${total} bars left"><i class="${tag}" style="width:${done}%"></i></div>`
        + `<span class="dim">hold ${left}/${total} bars left · ${num(p.age_hours, 1)}h ago</span>`
        + `</div>`;
    }).join("");
    posWrap.innerHTML = open || `<div class="dim pad">no open paper positions</div>`;
  }

  // ---- per-asset trigger coverage (honest none / blocked) ----
  const assetWrap = $("#book-assets");
  if (assetWrap) {
    // "UNPROVEN" / "SHADOW" / "NONE" were three labels with no explanation
    // (user report 2026-11-27). Each is a DIFFERENT fact and the tile now
    // says which: a shadow sleeve has a live reading the gate rejected, an
    // untested asset was never in the scan's search space, and a warming
    // sleeve is waiting for real Binance tape (never simulated).
    const rowsA = Object.entries(block.assets || {}).map(([sym, a]) => {
      const _st = String(a.state || "none");
      const _shadow = a.status === "shadow" || _st.startsWith("shadow:");
      // a shadow sleeve's `state` is contractually "unproven"; its live
      // reading arrives separately in `trigger`
      const _trig = String(a.trigger || _st.replace("shadow:", "") || "unproven");
      const state = (_shadow ? _trig : _st).toUpperCase();
      const c = state === "LONG" ? "acc-up" : state === "SHORT" ? "acc-dn"
        : state === "FLAT" ? "dim" : "acc-warn";
      const z = a.z == null ? ""
        : ` · live z ${Number(a.z).toFixed(2)} / ${Number(a.threshold || 2.5)}`;
      const kind = a.untested ? "UNTESTED — not in the scan"
        : _shadow ? "SHADOW — gate rejected, watched only"
        : a.status === "warming_up" ? "WARMING UP — needs real tape"
        : String(a.status || "none").toUpperCase();
      return `<div class="kv"><span>${sym}</span>`
        + `<b class="${c}">${state}</b>`
        + `<span class="dim">${esc(kind)}${z}</span>`
        + `<span class="dim" style="text-transform:none">${esc(a.note || "")}`
        + `${a.gate_note ? ` · <b>failed:</b> ${esc(a.gate_note)}` : ""}</span></div>`;
    }).join("");
    assetWrap.innerHTML = rowsA || `<div class="dim pad">no assets evaluated</div>`;
  }
}

/* One honest label for every surface: the sealed verdict. `issued` is kept
   true by the contract (the card always exists) so it can no longer be the
   thing the UI prints as "ISSUED". */
/* trend-interaction RULE (app/trend_gate.py). 2026-11-26: it no longer
   withholds a chasing call — it REVERSES the side, because the record says a
   fade wins 56.2 % (95 % CI 52.9-59.5, n=861) while the chase it replaces
   wins 46.0 % on 478. So this row must show three things a bare "stands"
   would hide: (1) whether THIS call was flipped, (2) the reading and the
   noise filter that decided it, and (3) the measurement behind the rule
   INCLUDING its losing fold — the panel is where "we know" and "we assume"
   are separated, and a flip printed without its fold-3 weakness would be a
   quiet lie. `blocked` still renders (TREND_GATE=abstain keeps the old
   semantic for A/B), but it is no longer the only action. */
function _trendGateRow(cyc) {
  const g = cyc?.trend_gate;
  if (!g || typeof g !== "object") return "";
  const me = g.measured || {};
  const fl = cyc?.trend_flip && typeof cyc.trend_flip === "object" && cyc.trend_flip.applied
    ? cyc.trend_flip
    : (g.flipped ? { from: g.side_in, to: g.side_out } : null);
  const tr = g.trend > 0 ? "up" : g.trend < 0 ? "down" : "no reading";
  const br = g.agrees === true
    ? `chasing ${fx((me.agrees_wr ?? 0) * 100, 1)}% (n=${me.agrees_n ?? "?"})`
    : g.agrees === false
      ? `fading ${fx((me.opposes_wr ?? 0) * 100, 1)}% (n=${me.opposes_n ?? "?"})`
      : "—";
  const state = fl
    ? `FLIPPED ${fl.from > 0 ? "BUY" : "SELL"}→${fl.to > 0 ? "BUY" : "SELL"}`
    : g.blocked ? "WITHHELD" : "stands";
  const cls = fl ? "acc-up" : g.blocked ? "acc-dn" : "dim";
  // the evidence block: policy win-rate, its CI, every fold (the losing one
  // is the honesty), net bps, and whether the noise filter was even armed.
  const ci = Array.isArray(me.flip_ci) && me.flip_ci.length === 2
    ? ` (95 % CI ${fx(me.flip_ci[0], 1)}-${fx(me.flip_ci[1], 1)})` : "";
  const folds = Array.isArray(me.flip_folds) && me.flip_folds.length
    ? ` · folds ${me.flip_folds.map((f) => fx(f * 100, 0)).join("/")}`
    : "";
  const net = me.flip_net_bps != null ? ` · net ${fmt(me.flip_net_bps, 1)} bps` : "";
  const ev = me.flip_n
    ? `<span class="dim" style="font-size:11px"> flip policy ${fx(me.flip_wr * 100, 1)}%${ci} on ${me.flip_n} vs baseline ${fx((me.baseline_wr ?? 0) * 100, 1)}%${folds}${net}</span>`
    : "";
  const nz = g.noise_filter === false
    ? `<span class="acc-warn" style="font-size:11px"> · noise filter unarmed: this cell's window σ is not measured yet, so the raw lag-${g.lag_rows ?? 2} sign decided</span>`
    : (g.min_trend_bps != null
      ? `<span class="dim" style="font-size:11px"> · trend ${fmt(g.trend_bps, 1)} bps vs ${fmt(g.min_trend_bps, 1)} bps noise floor (${fmt(g.window_sigma_bps, 0)} bps σ)</span>`
      : "");
  return `<div class="kv"><span>trend rule ${g.lag_rows ? `(${g.lag_rows} dec)` : ""}</span><b class="${cls}">${state} · trend ${tr} · ${br}<span class="dim" style="font-size:11px"> ${esc(g.why || "")}</span>${ev}${nz}</b></div>`;
}
/* WHERE the printed P comes from. `p_side` is the logistic of the score in
   floor units — a CONFIDENCE PROXY, not a calibrated probability. The board's
   own calibration panel says the realised hit-rate is ~0.50 in every
   confidence bin (n=4534), so a card reading "P 1.00" is describing the
   score's size, not a win probability. Label it, and put the measured
   probability beside it whenever one exists — the two must never be
   confusable. */
function _pTag(cyc, res) {
  const p = cyc?.p_side ?? res?.p_side;
  if (p == null) return "";
  const pc = cyc?.p_correct ?? res?.p_correct;
  const src = cyc?.calib_source ? ` (${cyc.calib_source})` : "";
  const meas = pc != null
    ? ` · measured ${fmt(pc, 2)}${(cyc?.calib_n ?? res?.calib_n) != null ? ` n=${cyc?.calib_n ?? res?.calib_n}` : ""}`
    : " · measured —";
  return ` · P ${fmt(p, 2)} score-units${src}${meas}`;
}
/* slot-status CSS classes are a fixed vocabulary; the status STRINGS are not
   (a stored-but-untested key must not render as the same grey as "no key"). */
function _slotCls(st) {
  const t = String(st || "").toLowerCase();
  if (t === "connected") return "connected";
  if (t === "disabled") return "disabled";
  if (t === "rate-limited") return "rate-limited";
  if (t.startsWith("key stored")) return "untested";
  if (t === "error") return "error";
  return "disconnected";
}
function verdictOf(cyc, res) {
  const v = cyc?.verdict || (cyc?.provisional ? "PREVIEW" : "");
  if (v === "ISSUE") return "ISSUED";
  if (v === "WATCH") return "WATCH";
  if (v === "NO CALL") return "NO CALL";
  if (v === "PREVIEW") return "PREVIEW";
  if (res?.gate_ok === true) return "ISSUED";
  if (res?.gate_ok === false) return "NO CALL";
  return "NO CALL";
}
/* issue 9 — sideways/choppy market caution: Kaufman efficiency ratio on
   the card's own tape. ER ≈ |net move| / Σ|Δ| — near 0 = pure chop
   (whipsaw), near 1 = clean trend.

   Bug fixed 2026-11-26: the threshold used to be ABSOLUTE (ER < 0.20 =
   caution). ER is not scale-free in the sample size — for a pure random
   walk E[ER(n)] = 1/sqrt(n). At the 60 bars this function looks at, a
   random walk averages ER 0.13, which is *below* 0.20, so the "sideways"
   flag fired on ~78 % of tapes that had no trend AND no range. Measured:

     n= 20  mean ER 0.232   P(ER<0.20) = 50.0 %
     n= 60  mean ER 0.129   P(ER<0.20) = 78.1 %
     n=120  mean ER 0.093   P(ER<0.20) = 91.7 %
     n=240  mean ER 0.065   P(ER<0.20) = 98.6 %

   A caution flag that fires on three quarters of all conditions carries no
   information. The threshold is now RELATIVE to the random-walk baseline
   for the number of bars actually used: cautious only when ER falls well
   below what noise alone would produce. */
function chopInfo(tf) {
  try {
    const bars = S.tfBars?.[tf] || [];
    const n = Math.min(bars.length, 60);
    if (n < 20) return null;
    const win = bars.slice(-n);
    let gross = 0;
    for (let i = 1; i < win.length; i++) {
      const d = Math.abs(win[i].close - win[i - 1].close);
      if (isFinite(d)) gross += d;
    }
    const net = Math.abs(win[win.length - 1].close - win[0].close);
    if (!(gross > 0)) return null;
    const er = net / gross;
    // random-walk baseline for this many bars
    const base = 1 / Math.sqrt(n);
    // cautious only when the tape is meaningfully less efficient than noise
    const level = er < base * 0.5 ? "severe" : er < base * 0.8 ? "chop" : null;
    return { er, base, level, n, rel: base > 0 ? er / base : 1 };
  } catch { return null; }
}
function chopChip(tf) {
  // §1.3/§6: the 5-state regime engine is server truth — computed from
  // THIS asset's tape (never BTC's) and stamped on every TF card.
  // Yellow caution for the sideways states, red for the shock state.
  try {
    const r = S.snap?.assets?.[S.symbol]?.tfs?.[tf]?.regime;
    if (r && r.state) {
      const conf = (r.confidence != null) ? ` · conf ${Number(r.confidence).toFixed(2)}` : "";
      const title = esc(`${r.state} — ${r.label}. ER ${num(r.er_fast)}, vol×${num(r.vol_ratio)}${conf} (5-state regime engine, this asset's tape)`);
      if (r.state === "CHAOTIC_LIQUIDATION_SHOCK")
        return `<span class="chip-err" title="${title}">🔴 LIQUIDATION SHOCK — no entries</span>`;
      if (r.state === "MEAN_REVERTING_RANGE")
        return `<span class="chip-warn" title="${title}">⚠ SIDEWAYS — trade with caution</span>`;
      if (r.state === "VOLATILE_RANGE")
        return `<span class="chip-warn" title="${title}">⚠ VOLATILE RANGE — whipsaw risk</span>`;
      if (r.state === "IMPULSE_BULL")
        return `<span class="chip-ok" title="${title}">📈 impulse bull${conf}</span>`;
      if (r.state === "IMPULSE_BEAR")
        return `<span class="chip-err" title="${title}">📉 impulse bear${conf}</span>`;
    }
  } catch { /* regime optional */ }
  // fallback: local Kaufman ER chip when the server regime is absent
  const c = chopInfo(tf);
  if (!c || !c.level) return "";
  return `<span class="chip-warn" title="Kaufman efficiency ratio ${c.er.toFixed(2)} vs ${c.base.toFixed(2)} expected from noise over ${c.n} bars (${(c.rel * 100).toFixed(0)}% of random-walk efficiency) — price is going nowhere faster than chance; whipsaw risk high"
    >⚠ SIDEWAYS${c.level === "severe" ? " (SEVERE)" : ""} — trade with caution</span>`;
}
function renderHero() {
  const tf = S.focusTf;
  const a = S.snap?.assets?.[S.symbol] || {};
  const cyc = a.tfs?.[tf];
  const res = S.snap?.systems?.[`${S.symbol}|${tf}`] || {};
  const m = TF_META[tf];
  // Bug fixed 2026-11-26: the header badge said "▲ BULLISH*" while the chip
  // right next to it said "📉 impulse bear · conf 0.10" - two panels, two
  // different regime computations. The chip is `chopChip()`, which re-derives
  // the regime locally; the engine computes its own on the TF tape and marks
  // the disagreement in `regime_conflict`.
  //
  // FIXED AGAIN 2026-11-26 (second pass, same day): the first fix made the
  // disagreement a NO CALL *everywhere*, which answered a contradiction with
  // an ABSTENTION — the one thing the user forbade, applied to a window whose
  // side was already measured, sealed and graded. The disagreement now shows
  // as a banner beside the real side: the hero prints the side the engine
  // certified (flipped by app/trend_gate.py when it chases the impulse), and
  // says plainly that routing is held back until the tape agrees with
  // itself. Never silence a call to resolve a contradiction; explain it.
  const conflict = res.regime_conflict || cyc?.regime_conflict || null;
  const dir = cyc?.direction ?? res.direction ?? 0;
  // Backend now always publishes a card (provisional when no cycle is
  // locked yet); these fallbacks keep the hero honest even mid-boot.
  const issued = cyc?.issued ?? res.issued ?? false;
  // ISSUED means "cleared the issue floor" — the analysis and the card now
  // read the same gate, so the hero can never advertise a call the recheck
  // rejects. `verdict` is the sealed answer written with the prediction lock.
  const gateOk = cyc?.gate_ok ?? res.gate_ok ?? issued;
  const prov = !!cyc?.provisional;
  const verdict = verdictOf(cyc, res);
  const lockedId = cyc?.prediction_lock?.id || res?.prediction_lock?.id || "";
  const heroEntry = cyc?.entry ?? res.entry ?? res.lock?.price ?? null;
  const heroSl = cyc?.sl ?? res.sl ?? null;
  const weak = String(res?.ladder?.source || "").includes("inside noise floor")
    || Number(res?.confidence ?? 0) < 0.05;
  const ttl = $("#hero-title");
  if (ttl) {
    ttl.innerHTML = `${tf.toUpperCase()} · <b style="color:var(--cyan)">${m.mins}m</b> window
      <span class="dim">${S.symbol} · sub-timeframes ${m.tps.join(" / ")}m</span>`;
  }
  const badge = $("#hero-badge");
  if (badge) {
    badge.className = "hero-badge " + (dir > 0 ? (weak ? "flat" : "up") : dir < 0 ? (weak ? "flat" : "down") : "flat");
    // A BADGE MAY NOT CONTRADICT ITS OWN CHIP (live defect 2026-11-27).
    // The headline read "▲ BULLISH*" while the chip beside it read
    // "📉 impulse bear · conf 0.20", and nothing on the card explained it.
    // When they disagree the badge now prints BOTH facts in one sentence —
    // the certified side first (it is the call, and it is graded) and the
    // regime read second — instead of two numbers that fight.
    const _sideTxt = dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO CALL";
    const _clash = conflict
      ? `<span class="chip-err" title="the 5-state regime on this timeframe's tape is ${esc(conflict.state)} (conf ${num(conflict.regime_conf)}) while the certified side is ${dir > 0 ? "BUY" : "SELL"}. The call stands and is graded; only ORDER ROUTING is held back until the tape agrees with itself. A withheld call is not a safer call — it is an ungraded one">⚠ ${esc(conflict.state)} (conf ${num(conflict.regime_conf)}) OPPOSES THIS ${dir > 0 ? "BUY" : "SELL"} · call stands and is graded, routing held</span>`
      : "";
    badge.innerHTML = `${_sideTxt}${weak && dir ? "*" : ""}<small>${S.symbol} · ${tf.toUpperCase()} ${m.mins}m · conf ${fx(res.confidence ?? 0, 3)} · ${verdict}${cyc?.tier ? ` · ${cyc.tier}` : ""}${_pTag(cyc, res)}${lockedId ? ` · lock ${lockedId.slice(-6)}` : ""}</small>${conflict ? "" : chopChip(tf)}${_clash}`;
  }
  const stats = $("#hero-stats");
  if (stats) {
    stats.innerHTML = `
      <div class="kv"><span>call</span><b class="${dir > 0 ? (weak ? "flat" : "up") : dir < 0 ? (weak ? "flat" : "down") : "flat"}">${dir > 0 ? "▲ BULL" : dir < 0 ? "▼ BEAR" : "● FLAT"}${weak && dir ? "*" : ""}${conflict ? ' <span class="acc-warn" style="font-size:11px">· ungraded-free: no order routed</span>' : ""}</b></div>
      ${conflict ? `<div class="kv"><span>regime disagreement</span><b class="acc-warn">the 5-state chip reads ${esc(conflict.state)} (conf ${num(conflict.regime_conf)}) against this ${dir > 0 ? "BUY" : "SELL"} — the side stands and is graded, only routing is held</b></div>` : ""}
      <div class="kv"><span>confidence</span><b>${fx(res.confidence ?? 0, 3)} · ${verdict}${res.issue_note ? ` <span class="dim" style="font-size:11px">${esc(res.issue_note)}</span>` : ""}${res.cell_certainty != null ? ` <span class="dim" style="font-size:11px">· certainty ${fx(res.cell_certainty, 3)} (n=${res.cell_certainty_n ?? "?"})</span>` : ""}</b></div>
      ${res.cell_win_rate_net != null ? `<div class="kv"><span>real win rate</span><b>${fx(res.cell_win_rate_net * 100, 1)}% after ${fx(res.cell_round_trip_bps ?? 0, 1)} bps RT <span class="dim" style="font-size:11px">(gross ${fx((res.cell_win_rate ?? 0) * 100, 1)}%)</span></b></div>` : ""}
      <div class="kv"><span>prediction lock</span><b class="${cyc?.prediction_lock?.locked ? "acc-up" : "dim"}">${cyc?.prediction_lock?.locked ? `🔒 ${esc(cyc.prediction_lock.side || "")} · ${esc(String(cyc.prediction_lock.id || "").slice(-8))}` : "not sealed yet"}</b></div>
      ${_trendGateRow(cyc)}
      ${_pWinRow(res)}
      <div class="kv"><span>strength (voter agreement)</span><b>${fx(cyc?.strength ?? res.strength ?? 0, 2)}</b></div>
      <div class="kv"><span>measured win-rate (this conf)</span><b>${cyc?.winrate != null ? `${Math.round(cyc.winrate * 100)}% · n=${cyc.winrate_n}` : `collecting · n=${cyc?.winrate_n ?? 0}`}</b></div>
      ${cyc?.forecast?.price ? `<div class="kv"><span>${cyc.forecast.mode === "new" ? "🎯 FORECAST (target)" : "forecast if NEW"} ${cyc.forecast.minutes}m</span><b>${px(cyc.forecast.price)} (${cyc.forecast.move_bps} bp)</b></div>` : ""}
      <div class="kv"><span>cci adj</span><b class="${dir && (res.cci_adjusted ?? 0) && Math.sign(dir) !== Math.sign(res.cci_adjusted) ? "acc-warn" : ""}">${num(res.cci_adjusted, 4)}${dir && res.cci_adjusted && Math.sign(dir) !== Math.sign(res.cci_adjusted)
        ? ` <span class="dim" style="font-size:11px">· opposite sign to the ${esc(res.direction_source || "rule")} side — the side was set by a mandate/hold rule, the CCI is the raw formula vote</span>` : ""}</b></div>
      <div class="kv"><span>countdown</span><b>${cyc?.countdown_s != null ? `${Math.floor(cyc.countdown_s / 60)}m ${cyc.countdown_s % 60}s` : "—"}</b></div>
      <div class="kv"><span>${prov ? "entry (preview)" : "entry (locked contract)"}</span><b>${heroEntry ? px(heroEntry) : "—"}</b></div>
      <div class="kv"><span>price now</span><b>${a.price ? px(a.price) : "—"}</b></div>
      <div class="kv"><span>stop loss</span><b class="acc-dn">${heroSl ? px(heroSl) : "—"}${cyc?.sl_provisional ? ' <span class="dim" style="font-size:11px">prov</span>' : ""}${verdict === "NO CALL" && heroSl ? ' <span class="dim" style="font-size:11px">grading only — will not be traded</span>' : ""}</b></div>
      ${(() => {
        // The panel printed "TP:SL ratio 1.21:1" beside "TP 5.4bp / SL 3.4bp"
        // (= 1.59) and a forecast target 0.57 above the entry — three
        // numbers, three different ratios, no label saying which object each
        // belongs to. Compute the ratio from the SAME entry and stop that are
        // printed above and say which target it used.
        //
        // THE TARGET MUST BE ON THE CALL'S OWN SIDE (live defect 2026-11-27).
        // The hero took `cyc.forecast.price` FIRST, without asking whether it
        // sat on the side the card was calling. The reported card was a SELL
        // at 3,632 whose printed target was 3,649 — ABOVE its own entry —
        // because the forecast had been computed for the pre-flip BUY and the
        // trend rule then flipped the side. The card rendered an unexecutable
        // short with a "0.87:1" ratio. A target on the wrong side is not a
        // weak target, it is not a target at all: fall through to the next
        // source that IS on the call's side, and when none is, say so.
        if (!heroEntry || !heroSl) return `<div class="kv"><span>TP:SL ratio</span><b>—</b></div>`;
        const risk = Math.abs(heroEntry - heroSl);
        if (!(risk > 0)) return `<div class="kv"><span>TP:SL ratio</span><b>—</b></div>`;
        const _side = dir > 0 ? 1 : dir < 0 ? -1 : 0;
        const _ahead = (t) => t != null && Number.isFinite(+t) &&
          (_side === 0 || (_side > 0 ? +t > heroEntry : +t < heroEntry));
        const _cands = [
          ["forecast target", cyc?.forecast?.price],
          ["cycle target", cyc?.tp],
          ["contract TP", res.contract_tp],
          ["locked contract TP", (cyc?.level_contract || {}).tp],
        ];
        const _pick = _cands.find(([, v]) => _ahead(v));
        const _wrong = _cands.filter(([, v]) => v != null &&
          Number.isFinite(+v) && !_ahead(v));
        if (!_pick) {
          // Nothing on the call's side: the levels and the badge disagree.
          // Print the disagreement — never a made-up ratio, and never a
          // silent "—" that hides an incoherent card.
          const _label = _side > 0 ? "LONG" : _side < 0 ? "SHORT" : "CALL";
          const _seen = _wrong.length
            ? ` (${_wrong.map(([n, v]) => `${n} ${px(v)}`).join(", ")})`
            : "";
          return `<div class="kv"><span>TP:SL ratio</span><b class="acc-warn">NO TARGET ON THE ${_label} SIDE<span class="dim" style="font-size:11px"> — every published target sits on the wrong side of entry ${px(heroEntry)}${_seen}; nothing can be executed, so no ratio is claimed</span></b></div>`;
        }
        const [src, tgt0] = _pick;
        // THE FLOOR WIDENS THE TARGET, NEVER NARROWS THE STOP (user rule,
        // absolute). A sub-1:1 pair is REPAIRED, not demoted: the stop is
        // exactly as measured above, the target moves out to 1.0x the risk,
        // and the card prints BOTH numbers so the repair can never pass for
        // a measurement. This is also what keeps the mode honest — before
        // it, a flipped call could be drawn with a ratio under the floor.
        const _rr0 = Math.abs(tgt0 - heroEntry) / risk;
        const _floorHit = _rr0 < 1.0;
        const tgt = _floorHit
          ? +(heroEntry + _side * risk).toFixed(8)
          : tgt0;
        // recomputed from the target and the stop PRINTED ON THIS CARD, so
        // the number can never describe a different pair than the panel does
        const rr = Math.abs(tgt - heroEntry) / risk;
        const _warn = _wrong.length
          ? ` <span class="acc-warn" style="font-size:11px">· ignored ${_wrong.map(([n, v]) => `${n} ${px(v)} (wrong side)`).join(", ")}</span>`
          : "";
        const _floor = _floorHit
          ? ` <span class="dim" style="font-size:11px">· ${_rr0.toFixed(2)}:1 as published — target widened to the 1:1 floor (${px(tgt)}); stop left unchanged at ${px(heroSl)}</span>`
          : "";
        // WHEN this ratio is one the data measured, say so and show the
        // evidence that earned it. A target ratio with no provenance is
        // just a number; the user asked for a win rate and is owed the
        // number that was actually measured next to the ratio it came from.
        const _fc = cyc?.forecast || {};
        const _ev = _fc.rr_evidence || null;
        const _provenance = _ev
          ? ` <span class="acc-up" style="font-size:11px">· measured: ${_fc.rr_target}:1 target`
            + ` — holdout WR ${(_ev.holdout_wr * 100).toFixed(1)}%, EV ${_ev.holdout_ev > 0 ? "+" : ""}${_ev.holdout_ev.toFixed(4)}R, PF ${_ev.pf} over n=${_ev.n_holdout}</span>`
          : (_fc.rr_source
            ? ` <span class="dim" style="font-size:11px">· target ratio ${_fc.rr_target}:1 (${esc(_fc.rr_source)} — this cell has no measured ≥1:1 win rate)</span>`
            : "");
        return `<div class="kv"><span>TP:SL ratio</span><b>${rr.toFixed(2)}:1 <span class="dim" style="font-size:11px">(${src} ${px(tgt)} vs stop ${px(heroSl)} from entry ${px(heroEntry)})</span>${_warn}${_floor}${_provenance}</b></div>`;
      })()}`;
  }
  // TP ladder — the TF's own horizon ALWAYS first (user rule)
  // provisional card / missing cycles: fall back to the measured preview
  // rungs the analysis already published, so the ladder is never empty
  const tg = (cyc?.targets && cyc.targets.length) ? cyc.targets
    : ((res.tps || []).map(t => ({ k: t.k, minutes: t.minutes,
        level: t.level, status: "preview" })));
  const order = tg.map((_, i) => i);
  if (tg.length >= 4) order.unshift(order.pop());   // [3,0,1,2]
  const strip = $("#subtf-strip");
  if (strip) {
    strip.innerHTML = order.map(i => {
      const t = tg[i];
      if (!t) return "";
      const isTf = i === tg.length - 1;
      // FROZEN RUNG FALLBACK (audit §1.2): the canonical contract carries
      // the price of every rung from the moment the window locks, so a
      // not-yet-tracked TP2/TP3 shows its frozen level instead of "— OPEN".
      const _canon = cyc?.canonical || {};
      const _lvl = (t.level != null) ? t.level : (_canon[`tp${i + 1}`] ?? null);
      const _frozen = t.level == null && _lvl != null;
      const st = t.status === "hit" ? "HIT ✓"
        : t.status === "locked" ? "LOCK 🔒"
        : _frozen ? "FROZEN ❄" : "open";
      const cls = t.status === "hit" ? "acc-up" : (t.status === "locked" || _frozen) ? "acc-warn" : "dim";
      const _entry = cyc?.entry || res.lock?.price || a.price || 0;
      const _rel = (_lvl && _entry) ? ` <span class="dim">${((_lvl / _entry - 1) * 100).toFixed(2)}% from entry</span>` : "";
      return `<div class="kv"><span>${isTf ? `TF ${t.minutes}m — this window` : `TP${i + 1} · ${t.minutes}m sub-frame`}</span><b class="${cls}">${_lvl ? px(_lvl) : "—"} <span class="dim">${st}</span>${_rel}</b></div>`;
    }).join("") || `<div class="dim pad">waiting for the first lock…</div>`;
  }
  // CERTIFIED SLEEVE + PAPER BOOK — evidence, not a claim. The registry only
  // contains sleeves that passed app/edge_gate.py out of fold; anything else
  // appears as "none / blocked" so silence is never mistaken for a signal.
  const sl = (S.snap?.sleeves?.assets || {})[S.symbol] || null;
  const book = (S.snap?.sleeves?.portfolio || {});
  const pos = (book.open_positions || []).find(p => p.asset === S.symbol);
  const rep = (book.sleeves || []).find(r => r.asset === S.symbol && r.status !== "closed");
  const slBlock = $("#hero-sleeve");
  if (slBlock) {
    if (!sl || sl.status === "none") {
      // "none" had two very different meanings and the panel said neither:
      // the scan TESTED this asset and every candidate failed the gate, or
      // the scan never produced a candidate for it at all. The payload now
      // distinguishes them (sl.untested / sl.note), so the row says which.
      slBlock.innerHTML = `
        <div class="kv"><span>certified sleeve</span><b class="dim">${sl && sl.untested ? `UNTESTED — ${S.symbol}` : `none cleared the gate — ${S.symbol}`}</b></div>
        <div class="kv"><span>why</span><b class="dim">${esc((sl && sl.note) || "edge scan found no trigger clearing |t| ≥ 2.0 after friction")}</b></div>
        ${sl && sl.gate_note ? `<div class="kv"><span>detail</span><b class="dim">${esc(sl.gate_note)}</b></div>` : ""}`;
    } else {
      const isOpen = sl.status === "open";
      // A shadow sleeve's state arrives as "shadow:long|short|flat". It is a
      // READING of the trigger, never a position — the label says so instead
      // of the bare word UNPROVEN, which read as "the app is not working".
      const _shadow = sl.status === "shadow"
        || String(sl.state || "").startsWith("shadow:");
      const _rawState = String(sl.trigger
        || String(sl.state || "").replace("shadow:", ""));
      const live = pos ? (pos.direction > 0 ? "LONG" : "SHORT")
        : isOpen ? String(_rawState || "flat").toUpperCase()
        : _shadow ? `SHADOW · ${String(_rawState).toUpperCase()}`
        : String(sl.state || "unproven").toUpperCase();
      const stCls = pos ? (pos.direction > 0 ? "acc-up" : "acc-dn")
                        : (_rawState === "long" ? "acc-up" : _rawState === "short" ? "acc-dn" : "dim");
      const zTxt = sl.z == null
        ? "warming up — needs a real Binance tape (simulated bars are excluded from certified signals)"
        : `${Number(sl.z).toFixed(2)}σ / ${Number(sl.threshold).toFixed(1)}σ`
          + (sl.status === "shadow" ? ` · ${Math.abs(Number(sl.z)) > Number(sl.threshold) ? "trigger is FIRING now" : "below the trigger"} (shadow — never traded)` : "");
      const holdTxt = pos
        ? `${pos.hold_bars} bars`
        : `${sl.hold_hours}h`;
      slBlock.innerHTML = `
        <div class="kv"><span>certified sleeve ${isOpen ? "" : "(shadow)"}</span><b class="${stCls}">${live}</b></div>
        <div class="kv"><span>trigger (lead impulse → ${S.symbol})</span><b>${zTxt}</b></div>
        <div class="kv"><span>OOF net EV · PF · n</span><b>${Number(sl.net_ev_bps).toFixed(1)} bp · ${Number(sl.profit_factor).toFixed(2)} · ${sl.n}</b></div>
        <div class="kv"><span>corr-adj |t| (purged folds)</span><b>${Number(sl.t_effective).toFixed(2)}</b></div>
        <div class="kv"><span>hold</span><b>${holdTxt}</b></div>
        ${pos ? `<div class="kv"><span>entry / invalidation</span><b>${px(pos.entry_price)} / <span class="acc-dn">${px(pos.stop)}</span></b></div>` : ""}
        ${pos ? `<div class="kv"><span>size · vol-target ×</span><b>${(pos.size * 100).toFixed(1)}% · ${Number(pos.scale).toFixed(2)}×</b></div>` : ""}
        ${rep && rep.live_trades ? `<div class="kv"><span>paper ledger (${rep.live_trades} trades)</span><b class="${rep.live_ev_bps > 0 ? "acc-up" : "acc-dn"}">${Number(rep.live_ev_bps).toFixed(1)} bp · t ${Number(rep.live_t).toFixed(2)}</b></div>` : ""}
        ${rep && rep.realized_slippage_bps != null ? `<div class="kv"><span>realised slippage vs friction</span><b>${Number(rep.realized_slippage_bps).toFixed(1)} bp</b></div>` : ""}
        ${rep && rep.decay_warning ? `<div class="kv"><span>alpha decay</span><b class="acc-dn">live expectancy below half of research — monitor armed</b></div>` : ""}
        ${!isOpen ? `<div class="kv"><span>gate it failed</span><b class="dim">${esc(sl.gate_note || (sl.reasons || []).slice(0, 2).join("; ") || "not recorded")}</b></div>` : ""}
        <div class="kv"><span>note</span><b class="dim">${esc(sl.note || "")}</b></div>`;
    }
  }
  renderBook();
  const rn = $("#risk-note");
  if (rn) {
    const rk = cyc?.risk || {};
    rn.textContent = (rk.note
      ? `TP/SL from measured moves — ${rk.note}${rk.provisional ? " · levels provisional" : ""} · frozen at the lock; an emergency re-lock re-anchors the same bps at the new entry`
      : "TP/SL: measured-moves block pending (needs a locked cycle)")
      + (res.asset_nature ? ` · ${S.symbol} profile: ${res.asset_nature} · issue floor ${fmt(res.asset_floor, 3)}` : "")
      + (res.behavior?.action ? ` · behaviour: ${res.behavior.action}` : "");
  }
  // Frozen contract pair is explicit; measured TP rungs remain separate.
  const lines = [];
  const _canonL = cyc?.canonical || {};
  const _contract = cyc?.level_contract || {};
  // TP FALLBACK (report: "tp lines not showing in own chart"): `cyc.tp`
  // is null whenever the levels gate publishes the TP through the level
  // contract instead, so the chart lost its only contract anchor above
  // the price. Read the same three places the ladder panel reads.
  const _tpMain = [cyc?.tp, _contract.tp, _canonL.tp4, _canonL.tp3]
    .map(Number).find(v => Number.isFinite(v) && v > 0) ?? null;
  const _entryL = [cyc?.entry, _contract.entry, res.lock?.price, a.price]
    .map(Number).find(v => Number.isFinite(v) && v > 0) ?? null;
  const _slL = [cyc?.sl, _contract.sl].map(Number)
    .find(v => Number.isFinite(v) && v > 0) ?? null;
  if (_entryL) lines.push({ price: _entryL, color: "#c9d1d9", title: "ENTRY", contract: true });
  if (_tpMain) lines.push({ price: _tpMain, color: "#26a69a", title: "TP", contract: true });
  if (_slL) lines.push({ price: _slL, color: "#f0385a", title: "SL", contract: true });
  // the intermediate rungs stay annotations (they must not squash candles)
  [1, 2, 3].forEach(k => {
    const lv = Number((cyc?.targets?.[k - 1]?.level) ?? _canonL[`tp${k}`]);
    if (Number.isFinite(lv) && lv > 0 && (!_tpMain || Math.abs(lv - _tpMain) > 1e-9))
      lines.push({ price: lv, color: "rgba(38,166,154,.70)", title: `TP${k}`, dash: true });
  });
  const mgi = res.micro;
  if (mgi?.available) {
    // only clusters within ±2% of price: a far cluster used to widen
    // the chart range and squash the candles + TP/SL into a corner
    const _ref = a.price || cyc?.price || 0;
    const near = p => _ref > 0 && Math.abs(p - _ref) / _ref <= 0.02;
    if (mgi.nearest_up?.price && near(mgi.nearest_up.price))
      lines.push({ price: mgi.nearest_up.price, color: "#a855f7",
                   title: `LIQ↑ ${Math.round(mgi.nearest_up.bps)}bp`, dash: true });
    if (mgi.nearest_dn?.price && near(mgi.nearest_dn.price))
      lines.push({ price: mgi.nearest_dn.price, color: "#a855f7",
                   title: `LIQ↓ ${Math.round(mgi.nearest_dn.bps)}bp`, dash: true });
  }
  const _canon = cyc?.canonical || {};
  order.forEach(i => {
    const t = tg[i];
    if (!t) return;
    // frozen rung from the canonical contract when the target itself has
    // no level yet (pending TP2/TP3) — one ladder, drawn everywhere
    const lvl = (t.level != null) ? t.level : (_canon[`tp${i + 1}`] ?? null);
    if (lvl == null) return;
    const isTf = i === tg.length - 1;
    const frozen = t.level == null;
    lines.push({
      price: lvl,
      color: t.status === "hit" ? "#22c98a" : (t.status === "locked" || frozen) ? "#c9a227" : "#00d4f0",
      title: isTf ? `TF${t.minutes}m` : `TP${i + 1}${t.status === "hit" ? " ✓" : (t.status === "locked" || frozen) ? " 🔒" : ""}`,
      dash: t.status !== "locked" && !frozen,
    });
  });
  S.charts.hero?.setLines(S.showLines === false ? [] : lines);
  bnDrawLines(lines);
}

/* TP/SL/entry lines on the Binance LINE chart (the TV widget cannot
   take external lines; OWN + Binance-line can) */
function bnDrawLines(lines) {
  const inst = Bn.main; if (!inst) return;
  if (!lines) {   // rebuild from the focus cycle
    const cyc = S.snap?.assets?.[S.symbol]?.tfs?.[S.focusTf];
    lines = [];
    if (Number.isFinite(cyc?.entry)) lines.push({ price: cyc.entry, color: "#e8e8f0", title: "ENTRY", contract: true });
    if (Number.isFinite(cyc?.tp)) lines.push({ price: cyc.tp, color: "#22c98a", title: "TP", contract: true });
    if (Number.isFinite(cyc?.sl)) lines.push({ price: cyc.sl, color: "#f0385a", title: "SL", contract: true });
    for (const t of (cyc?.targets || [])) if (t.level)
      lines.push({ price: t.level, color: "#c9a227", title: `TP${t.k}` });
  }
  const visible = S.showLines === false ? [] : (lines || []).filter(L => Number.isFinite(L.price));
  inst.levelValues.splice(0, inst.levelValues.length,
    ...visible.filter(L => L.contract).map(L => L.price));
  for (const l of inst._lines || []) { try { inst.candles.removePriceLine(l); } catch {} }
  inst._lines = [];
  for (const L of visible)
    inst._lines.push(inst.candles.createPriceLine({ price: L.price,
      color: L.color || "#c9a227", lineWidth: 1, lineStyle: 2,
      axisLabelVisible: true, title: L.title || "" }));
  try { inst.chart.priceScale("right").applyOptions({ autoScale: true }); } catch {}
}

function bindAllTf() {
  const b = $("#btn-all-tf");
  if (b && !b.dataset.wired) {
    b.dataset.wired = "1";
    b.addEventListener("click", () => {
      S.tfFocus = false;
      try { localStorage.setItem("tfFocus", "0"); } catch (e) { /* private */ }
      renderDashboardMode();
      toast("all four timeframes shown", "ok");
    });
  }
}

function refreshActiveView(force) {
  if (S.view === "dashboard") {
    renderWhy(); renderQuads();
    try { renderDashboardMode(); } catch (e) { console.error("mode", e); }
    try { renderBranches(); } catch (e) { console.error("branches", e); }
    try { renderMicro(); } catch (e) { console.error("micro", e); }
    if (!force || !S._sysLoaded || Date.now() - S._sysLoaded > 30000) {
      S._sysLoaded = Date.now();
      loadSystemPanel().catch(() => {});
    }
    return;
  }
  // non-dashboard views re-FETCH their APIs — throttle snapshot-driven
  // refreshes to every 30s so the UI never stalls behind repeated loads
  if (!force && Date.now() - (S.lastViewFetch || 0) < 30000) return;
  S.lastViewFetch = Date.now();
  if (S.view === "formulas") { ensureExplainPanel("formulas"); loadFormulas(); loadExplain("formulas"); }
  else if (S.view === "emotion") { ensureExplainPanel("emotion"); loadEmotion(); loadExplain("emotion"); }
  else if (S.view === "physics") { ensureExplainPanel("physics"); loadPhysics(); loadExplain("physics"); }
  else if (S.view === "news") { ensureExplainPanel("news"); loadNews(); loadExplain("news"); }
  // (news refresh button wired below)
  else if (S.view === "brains") loadBrains();
  else if (S.view === "agents") loadAgents();
  else if (S.view === "mind") loadMind(true);
  else if (S.view === "history") loadHistory();
  else if (S.view === "settings") loadSettings();
}

/* ═══════════ dashboard: quadrants ═══════════ */
/* mini candle chart for the TF cards (doc C-5): candles + price lines
   only — no volume, no fan, no noise line; scroll/scale disabled.
   Keeps the setData/setLines/tick API the app already calls. */
function quadChart(el, bucketSec = 60) {
  const LW = window.LightweightCharts;
  const noop = { setData() {}, setLines() {}, tick() {} };
  if (!LW || !el) return noop;
  const chart = LW.createChart(el, {
    layout: { background: { color: "transparent" }, textColor: "#cbd5e1", fontSize: 9 },
    grid: { vertLines: { color: "rgba(140,140,170,.06)" }, horzLines: { color: "rgba(140,140,170,.06)" } },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: "rgba(140,140,170,.25)" },
    handleScroll: false, handleScale: false,
    width: el.clientWidth || 170, height: el.clientHeight || 92,
  });
  const levelValues = [];
  const candles = chart.addLineSeries({ color: "#38bdf8", lineWidth: 1,
    autoscaleInfoProvider: levelAutoscaleProvider(levelValues),
    priceLineVisible: false, lastValueVisible: true });
  new ResizeObserver(() => {
    const w = el.clientWidth, h = el.clientHeight;
    if (w > 20 && h > 20) { try { chart.applyOptions({ width: w, height: h }); } catch {} }
  }).observe(el);
  const inst = { chart, candles, last: null, _lines: [], levelValues,
    setData(bars) {
      if (!bars?.length) {
        candles.setData([]); inst.last = null;
        try { chart.timeScale().fitContent(); } catch {}
        return;
      }
      // a hidden slot can be created 0×0 (clientHeight 0): re-measure on
      // every paint so a card that became visible always gets a real size
      const w = el.clientWidth, h = el.clientHeight;
      if (w > 0 && h > 0 && (chart._w !== w || chart._h !== h)) {
        chart._w = w; chart._h = h;
        try { chart.applyOptions({ width: w, height: h }); } catch {}
      }
      candles.setData(bars.map(b => ({ time: lt(b.time), value: b.close })));
      const lb = bars[bars.length - 1];
      inst.last = { time: lb.time, open: lb.open, high: lb.high, low: lb.low, close: lb.close };
      try { chart.timeScale().scrollToRealTime(); } catch {}
    },
    setLines(lines) {
      const visible = (lines || []).filter(L => Number.isFinite(L.price));
      levelValues.splice(0, levelValues.length,
        ...visible.filter(L => L.contract).map(L => L.price));
      for (const l of inst._lines) { try { candles.removePriceLine(l); } catch {} }
      inst._lines = [];
      for (const L of visible)
        inst._lines.push(candles.createPriceLine({ price: L.price, color: L.color || "#c9a227",
          lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: L.title || "" }));
      try { chart.priceScale("right").applyOptions({ autoScale: true }); } catch {}
    },
    tick(p) {
      if (!inst.last || !(p > 0)) return;
      const bt = Math.floor(serverNow() / bucketSec) * bucketSec;
      if (bt > inst.last.time)   // bucket roll: advance, never freeze
        inst.last = { time: bt, open: inst.last.close, high: Math.max(inst.last.close, p),
                      low: Math.min(inst.last.close, p), close: p };
      else { const l = inst.last;
        l.close = p; l.high = Math.max(l.high, p); l.low = Math.min(l.low, p); }
      const l = inst.last;
      try { candles.update({ time: lt(l.time), value: l.close }); } catch {}
    } };
  return inst;
}

function buildQuads() {
  const grid = $("#quad-grid"); grid.innerHTML = "";
  TF.forEach(tf => {
    const m = TF_META[tf];
    const q = document.createElement("div");
    q.className = "quad" + (tf === S.focusTf ? " focus" : "");
    q.id = "q-" + tf;
    q.innerHTML = `
      <div class="q-head">
        <span class="q-tf">${tf.toUpperCase()} · ${m.mins}m <small>TP ${m.tps.join("/")}</small></span>
        <span class="q-clock"><svg viewBox="0 0 36 36">
          <circle class="ring-bg" cx="18" cy="18" r="15.5"/>
          <circle class="ring-fg" cx="18" cy="18" r="15.5"/></svg><b class="q-time">—</b></span>
      </div>
      <div class="q-chart" id="qc-${tf}"></div>
      <div class="q-tv hidden" id="qtv-${tf}"></div>
      <div class="q-bn hidden" id="qbn-${tf}"></div>
      <div class="q-price" id="qp-${tf}">—</div>
      <div class="q-dir flat" id="qd-${tf}">—</div>
      <div class="q-tps" id="qt-${tf}"></div>
      <div class="q-foot"><span id="qf-${tf}"></span><span id="qa-${tf}"></span></div>`;
    grid.appendChild(q);
    q.addEventListener("click", () => { if (S.focusTf !== tf) setFocusTf(tf); });
    S.charts[tf] = quadChart($("#qc-" + tf), (tf === "tf3" || tf === "tf4") ? 300 : 60);
  });
  // main OWN chart is the PriceChart renderer (drawMainChart)
  // into the corner under a multi-hour trend
  renderQuads(); loadCandles();
}
function renderQuads(opts = {}) {
  // heavy=false is the 1 Hz tick path: text cells + chart ticks ONLY.
  // Tables (calls/ledger/hindsight/meta) rebuild on snapshots/GRCs —
  // rebuilding them every second was the "laggy chart" complaint.
  const heavy = opts.heavy !== false;
  const a = S.snap?.assets?.[S.symbol] || {};
  TF.forEach(tf => {
    const cyc = a.tfs?.[tf];
    const res = S.snap?.systems?.[`${S.symbol}|${tf}`];
    // same rule as the hero: the quick bar shows the side the engine
    // certified. A regime disagreement only holds ROUTING back — it never
    // blanks the side (that was an abstention wearing a chip's clothes).
    const dir = cyc?.direction ?? res?.direction ?? 0;
    const _cf = res?.regime_conflict || cyc?.regime_conflict || null;
    const qd = $("#qd-" + tf);
    if (qd) {
      qd.className = "q-dir " + (dir > 0 ? "up" : dir < 0 ? "down" : "flat");
      const v = verdictOf(cyc, res);
      qd.innerHTML = `${dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO CALL"}
        <span class="iss">${cyc?.provisional
          ? "PREVIEW · no lock yet"
          : `${v}${cyc?.tier ? ` · ${cyc.tier}` : ""}${_pTag(cyc, res)}${
            _cf ? ` · <span class="acc-warn" title="the 5-state regime on this tape disagrees with the certified side: the call is published and graded, only order routing is held back">⚠ no routing</span>` : ""
          }`}</span>`;
    }
    const _rk = cyc?.risk || {};
    $("#qf-" + tf).textContent = res
      ? `CCI ${fmt(res.cci_adjusted, 4)} · conf ${fmt(res.confidence, 3)}`
        + (_rk.tp_bps != null ? ` · TP +${fmt(_rk.tp_bps, 0)}bp / SL −${fmt(_rk.sl_bps, 0)}bp` : "")
      : "";
    // THE ANCHOR SENTENCE (user report: "I got tp of 551 on BNB when the
    // market was on 556"). The TP/SL belong to the LOCKED entry, not to
    // spot, and that is correct — but the card printed only the levels, so
    // a target that looked "behind" the market had no explanation on
    // screen. Print the anchor, the live price and the drift beside them.
    const _qdrift = (() => {
      const dr = res?.anchor_drift_bps ?? cyc?.anchor_drift_bps;
      const lp = res?.live_price ?? cyc?.live_price;
      const ent = cyc?.entry ?? res?.entry;
      if (dr == null || !(lp > 0) || !(ent > 0)) return "";
      const big = Math.abs(dr) > 25;
      const stale = res?.tp_already_reached ?? cyc?.tp_already_reached;
      return `anchor ${px(ent)} · now ${px(lp)} <b class="${big ? "acc-warn" : "dim"}">`
        + `${dr >= 0 ? "+" : ""}${fmt(dr, 0)}bp</b> since the lock`
        + (stale ? ` <b class="acc-warn">· TP already printed — behind spot, not ahead</b>` : "");
    })();
    if (_qdrift && res) {
      const qf = $("#qf-" + tf);
      if (qf) qf.insertAdjacentHTML("afterend",
        `<span class="q-drift" style="font-size:11px" title="Levels are measured from the LOCKED entry, never from a later print. A target is where the trade was opened plus its measured excursion."> · ${_qdrift}</span>`);
    }
    const ag = res?.agent;
    const _agMode = S.snap?.agents?.mode || S.snap?.agents?.active_slot || "";
    $("#qa-" + tf).textContent = ag && ag.source && ag.source !== "none"
      ? `AI ${ag.direction > 0 ? "+" : ag.direction < 0 ? "−" : "0"} (${ag.source})`
      : (_agMode ? `AI ${_agMode} · no reply` : "AI — not configured");
    const tp = $("#qt-" + tf);
    if (tp) {
      const _qtg = cyc?.targets || [];
      const _qord = _qtg.map((_, i) => i);
      if (_qtg.length >= 4) _qord.unshift(_qord.pop());
      tp.innerHTML = _qord.map(i => {
        const t = _qtg[i];
        if (!t) return "";
        const isTf = i === _qtg.length - 1;
        const lbl = isTf ? `TF${t.minutes}m` : `TP${i + 1}`;
        return `<span class="tp ${t.status === "hit" ? "hit" : t.status === "locked" ? "locked" : ""}${isTf ? " tp-horizon" : ""}">${lbl}${t.status === "hit" ? " ✓" : t.status === "locked" ? " 🔒" : ""} ${t.level ? px(t.level) : ""}</span>`;
      }).join("") + (cyc?.sl ? `<span class="tp sl">SL ${px(cyc.sl)}</span>` : "");
    }
    // price line
    const pr = S.prices[S.symbol] ?? a.price;
    const chg = a.window_change_pct ?? 0;
    $("#qp-" + tf).innerHTML = pr != null
      ? `${px(pr)} <span class="chg ${chg >= 0 ? "up" : "down"}">${chg >= 0 ? "+" : ""}${fmt(chg, 2)}%</span>` : "—";
    // Frozen contract pair scales with the price series; measured rungs stay distinct.
    const lines = [];
    if (Number.isFinite(cyc?.entry)) lines.push({ price: cyc.entry, color: "#c9d1d9", title: "ENTRY", contract: true });
    if (Number.isFinite(cyc?.tp)) lines.push({ price: cyc.tp, color: "#26a69a", title: "TP", contract: true });
    if (Number.isFinite(cyc?.sl)) lines.push({ price: cyc.sl, color: "#f0385a", title: "SL", contract: true });
    // liquidation clusters (measured @forceOrder buckets) as purple overlays
    // (liquidation-cluster overlays live on the hero chart only — on
    // the small quad cards they were two more colliding labels)
    // quad cards show ONLY the horizon TP + SL lines: every sub-target
    // line on a 90 px card was an unreadable label stack (the values
    // themselves are in the text row under the card and the calls table)
    const _hz = (cyc?.targets || [])[(cyc?.targets || []).length - 1];
    if (_hz?.level) lines.push({
      price: _hz.level, color: "#c9a227",
      title: `TF${_hz.minutes}m rung`, dash: false,
    });
    if (heavy) S.charts[tf]?.setLines(lines);
  });
  if (!heavy) return;
  renderRail();
  try { renderCalls(); } catch (e) { console.error("calls", e); }
  try { renderLedger(); renderHindsight(); renderMeta(); } catch (e) { console.error("ledger", e); }
}
/* ═══════ AUTO-PREDICT toggle (dashboard chip + settings row) ═══════ */
async function loadAutoPredict() {
  try {
    const st = await api("/api/settings");
    // MANUAL IS THE DEFAULT (user directive): auto only when the
    // server says the user explicitly enabled it — undefined/off
    // renders OFF.  Only the user flips this chip.
    const on = st?.settings?.auto_predict === true;
    S.autoPredict = on;
    const chip = $("#auto-chip");
    if (chip) {
      chip.textContent = on ? "AUTO: ON" : "AUTO: OFF";
      chip.className = "btn-analyse btn-auto " + (on ? "auto-on" : "auto-off");
      chip.title = on
        ? "AUTO-PREDICT on (you enabled it) — analysis re-runs by itself every ~45 s. Click for MANUAL mode (▶ ANALYSE only)."
        : "MANUAL mode (default) — press ▶ ANALYSE on the dashboard for a fresh full analysis. Only you can turn AUTO on.";
    }
    const row = $("#st-auto");
    if (row) row.textContent = on ? "ON — analysis runs itself (click to toggle)" : "OFF — manual ▶ ANALYSE only (default, click to toggle)";
  } catch { /* offline */ }
}
function bindAutoChip() {
  const chip = $("#auto-chip");
  if (chip) chip.addEventListener("click", async () => {
    const next = !(S.autoPredict === true);
    try {
      await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" },
                                   body: JSON.stringify({ auto_predict: next }) });
      toast(next ? "AUTO-PREDICT on — analysis runs itself (user-enabled)" : "MANUAL — press ▶ ANALYSE for a fresh run", "ok");
    } catch (e) { toast("toggle: " + e.message, "err"); }
    loadAutoPredict();
  });
  const row = $("#st-auto");
  if (row) row.addEventListener("click", () => chip && chip.click());
  loadAutoPredict();
}

/* ═══════ chart source: OWN charts ⇄ TradingView (proper toggle) ═══════ */
const TV_SYMS = { BTC: "BINANCE:BTCUSDT", PAXG: "BINANCE:PAXGUSDT",
                  ETH: "BINANCE:ETHUSDT", BNB: "BINANCE:BNBUSDT" };
const TV_INT = { tf1: "5", tf2: "15", tf3: "60", tf4: "240" };
function applyChartSrc() {
  const mode = S.chartSrc || "binance";   // binance | tv | own
  $$("#seg-chart .seg-btn[data-chart]").forEach(b =>
    b.classList.toggle("active", b.dataset.chart === mode));
  const own = $("#chart-own"), bn = $("#chart-binance"), wrap = $("#chart-tv");
  if (own) own.classList.toggle("hidden", mode !== "own");
  if (bn) bn.classList.toggle("hidden", mode !== "binance");
  if (wrap) wrap.classList.toggle("hidden", mode !== "tv");
  const tvSrc = (tf) => "https://s.tradingview.com/widgetembed/?symbol="
    + encodeURIComponent(TV_SYMS[S.symbol] || "BINANCE:BTCUSDT")
    + `&interval=${TV_INT[tf] || "5"}&theme=dark&style=1`
    + "&timezone=Etc%2FUTC&withdateranges=1&hide_side_toolbar=0"
    + "&allow_symbol_change=0&saveimage=0";
  const fill = (el, tf) => { if (!el) return; const src = tvSrc(tf);
    if (el.dataset.src !== src) { el.dataset.src = src;
      el.innerHTML = `<iframe src="${src}" style="width:100%;height:100%;border:0" allowtransparency="true" scrolling="no" allowfullscreen></iframe>`; } };
  const fb2 = $("#btn-fan");
  if (fb2) { fb2.classList.toggle("hidden", mode !== "own"); fb2.classList.toggle("active", !!S.fanOn); }
  if (mode === "tv" && wrap) fill(wrap, S.focusTf);
  if (mode === "binance") bnEnsure();
  if (mode === "own") drawMainChart();
  // The four TF cards ALWAYS draw the server tape (own series): that is
  // the only renderer that can carry the entry / TP / SL / horizon-rung
  // markers, and it cannot be blank. Previously the default BINANCE mode
  // swapped in `#qbn-<tf>`, whose chart was created while the slot was
  // display:none (clientHeight 0) and therefore never painted — the
  // reported blank mini sparklines. The big chart keeps the full
  // OWN / BINANCE / TV toggle.
  for (const tf of TF) {
    const o = $("#qc-" + tf),
          slot = $("#qtv-" + tf), bslot = $("#qbn-" + tf);
    if (o) o.style.display = "";
    if (slot) slot.classList.add("hidden");
    if (bslot) bslot.classList.add("hidden");
    const inst = S.charts[tf];
    if (inst?.chart && o) {
      const w = o.clientWidth, h = o.clientHeight;
      if (w > 0 && h > 0) {
        try { inst.chart.applyOptions({ width: w, height: h }); } catch (e) {}
        try { inst.chart.timeScale().fitContent(); } catch (e) {}
      }
    }
  }
}
function bindChartSrc() {
  try { S.chartSrc = localStorage.getItem("chartSrc") || "binance"; } catch { S.chartSrc = "binance"; }
  if (!["binance", "tv", "own"].includes(S.chartSrc)) S.chartSrc = "binance";
  try { S.fanOn = localStorage.getItem("fanOn") === "1"; } catch { S.fanOn = false; }
  try { S.showLines = localStorage.getItem("showLines") !== "0"; } catch { S.showLines = true; }
  const lb = $("#btn-lines");
  if (lb) { lb.classList.toggle("active", !!S.showLines);
    lb.addEventListener("click", () => { S.showLines = !S.showLines;
      try { localStorage.setItem("showLines", S.showLines ? "1" : "0"); } catch {}
      lb.classList.toggle("active", !!S.showLines);
      try { drawMainChart(); bnDrawLines(); } catch {} }); }
  const fb = $("#btn-fan");
  if (fb) { fb.classList.toggle("active", !!S.fanOn);
    fb.addEventListener("click", () => { S.fanOn = !S.fanOn;
      try { localStorage.setItem("fanOn", S.fanOn ? "1" : "0"); } catch {}
      fb.classList.toggle("active", !!S.fanOn); Own.fanKey = "__toggle";
      try { drawMainChart(); } catch {} }); }
  $$("#seg-chart .seg-btn[data-chart]").forEach(b => b.addEventListener("click", () => {
    S.chartSrc = b.dataset.chart;
    try { localStorage.setItem("chartSrc", S.chartSrc); } catch {}
    applyChartSrc();
  }));
  applyChartSrc();
}

/* ═══════ staged ANALYSE: button, timeline, result box ═══════ */
function bindAnalyse() {
  const b = $("#btn-analyse");
  if (!b) return;
  b.addEventListener("click", async () => {
    if (b.disabled) return;
    b.disabled = true;
    const sym = S.symbol;
    try {
      // §1.1 latest-wins: each click mints a monotonic request_id; the
      // server stamps it onto every analyse_* event and the WS handler
      // drops anything older than the id this cell currently owns
      S.analyseSeq = (S.analyseSeq || 0) + 1;
      const reqId = `${Date.now()}.${S.analyseSeq}`;
      const cellKey = `${sym}|${S.focusTf}`;
      S.analyseReqIds = S.analyseReqIds || {};
      S.analyseReqIds[cellKey] = reqId;
      const r = await api(`/api/analyse/${sym}?tf=${S.focusTf}&request_id=${encodeURIComponent(reqId)}`, { method: "POST" });
      if (!r.ok) { toast(r.error || "analysis unavailable", "err"); b.disabled = false; return; }
      // show the panel on the dashboard immediately — timestamps anchor
      // to the button press (started_at comes from the server)
      setView("dashboard");
      const panel = $("#analyse-panel");
      panel?.classList.remove("hidden");
      // issue 3: already_running/queued means a job is ALIVE — keep the
      // live stage rows instead of resetting them to "queued" (the old
      // reset wiped progress and looked like the click did nothing)
      if (r.already_running || r.queued) {
        $("#an-title").textContent = `ANALYSING ${sym} · ${S.focusTf.toUpperCase()}`;
        $("#an-sub").textContent = r.queued
          ? `queued (position ${r.queue_position || 1}) — waiting for the running job…`
          : "already running — following the live stages below…";
        resyncAnalyse();
        b.disabled = false;
        return;
      }
      const started = new Date(r.started_at * 1000);
      $("#an-title").textContent = `ANALYSING ${sym} · ${S.focusTf.toUpperCase()}`;
      $("#an-sub").textContent = `started ${started.toISOString().slice(11, 23)}Z — calculating…`;
      $("#an-stages").innerHTML = (r.stages || []).map((name, i) =>
        `<div class="an-row" id="anr-${name}">
           <span class="dim">${i + 1}</span><span class="an-name">${name.replace("_", " ")}</span>
           <span class="an-ms">…</span><span class="an-at">—</span>
           <span class="an-sum dim">queued</span></div>`).join("");
      const box = $("#an-result");
      if (box) { box.innerHTML = ""; box.classList.add("hidden"); }
      // REST fallback: if analyse_done never arrives (WS blip), the poll
      // below still completes the panel
      startAnalysePoll();
      loadCells();                         // this cell just armed its timer
      return;                              // stay disabled while it runs
    } catch (e) {
      if (e && e.body && e.body.error === "cell_locked") {
        // per-cell ANALYSE timer: THIS cell sleeps, others stay free
        const m = Math.floor((e.body.remaining_s || 0) / 60);
        const ss = Math.floor((e.body.remaining_s || 0) % 60);
        toast(`⏳ ${sym} ${S.focusTf.toUpperCase()} locked — ${m}m ${ss}s left on its ${e.body.minutes_left || m + 1}-min timer. Other cells still analyse.`, "err");
        loadCells();                       // re-render keeps it disabled
        return;                            // until the countdown expires
      }
      toast("analyse: " + e.message, "err");
    }
    b.disabled = false;                    // hard errors re-enable at once
  });
}

/* ═══════ HOLD POLICY cells: 16-card grid + per-cell timers ═══════ */
async function loadCells() {
  try {
    const d = await api("/api/cells");
    S.cells = d;
    renderCells();
  } catch { /* offline — keep last render */ }
}
function _tierCls(tier) {
  if (tier === "HOLD-OUT VERIFIED") return "tier-hold";
  if (tier === "HOLDOUT ROWS · ERA n<3") return "tier-rows";
  if (tier === "UNGRADEABLE / WARMUP") return "tier-warm";
  // a certified geometry that pays less than it risks is a violation of a
  // hard rule, not a soft "shadow" measurement — it must read as an error
  if (tier === "GEOMETRY BELOW 1:1 · NOT TRADEABLE") return "tier-err";
  // no signal at all is the more fundamental failure — the signal does not
  // predict a move, so no target ratio can repair it
  if (tier === "NO MEASURED EDGE · PATTERN NOT TRADEABLE") return "tier-err";
  return "tier-shadow";
}
function _cdText(ms) {
  if (ms <= 0) return null;
  const s = Math.ceil(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function renderCells() {
  const box = $("#cells-cards");
  if (!box || !S.cells) return;
  const cells = S.cells.cells || {};
  const now = Date.now() + (S.serverOffset || 0);
  $("#cells-meta").textContent =
    `${Object.keys(cells).length} cells · tiers: ` +
    Object.entries(S.cells.tiers || {}).map(([k, v]) => `${k} ×${v}`).join(" · ");
  box.innerHTML = Object.entries(cells).map(([key, c]) => {
    const [sym, tf] = key.split("|");
    const a = c.analyse || {};
    const untilMs = (a.locked_until || 0) * 1000;
    const left = untilMs - now;
    const locked = left > 0;
    const active = sym === S.symbol && tf === S.focusTf;
    // "n/a" with no reason is a dead end, not an answer (user report
    // 2026-11-27: "Why their is n/a?"). A cell with no certified win rate
    // is either UNGRADEABLE on its own history or has not been measured
    // yet, and those are different facts — say which, and why.
    const wr = c.win_rate != null ? `${(c.win_rate * 100).toFixed(1)}%` :
      (c.certified ? "—" :
        (c.status === "ungradeable" || c.tier === "UNGRADEABLE / WARMUP")
          ? `<span class="dim" title="${esc(c.why || "")}">no rate — ungradeable</span>`
          : `<span class="dim">no rate — not measured</span>`);
    // the number above is a GROSS first-touch hit rate — label it so it
    // can never be read as money
    const wrLabel = c.win_rate != null ? " <span class=\"dim\">gross</span>" : "";
    // ── the ratio the win rate BELONGS to. Until 2026-11-27 the cell
    // printed a 74 % win rate and never the payoff it was measured at
    // (0.15 x stop, i.e. 0.27:1), so the headline read as a good trade
    // when the certified geometry cannot pay for its own stop and the
    // live lock runs a repaired 1:1 whose hit rate was never measured.
    const _rrBad = c.geometry_feasible === false;
    const rrTxt = c.geometry_rr != null
      ? `<span class="${_rrBad ? "acc-dn" : "dim"}">${num(c.geometry_rr, 2)}:1${
          _rrBad ? ` — BELOW THE 1:1 FLOOR` : ""}</span>`
      : "";
    const conf = c.confidence != null ? `wilson lo ${(c.confidence * 100).toFixed(1)}%` :
      (c.tier === "UNGRADEABLE / WARMUP" ? "warmup — no certified WR" : "");
    const dir = c.live_direction > 0 ? `<span class="buy">▲ BUY</span>` :
      c.live_direction < 0 ? `<span class="sell">▼ SELL</span>` : `<span class="dim">● no live side</span>`;
    const era = c.era || {};
    const eraTxt = era.error ? "" :
      era.side ? `${era.in_position ? "IN" : "flat"} ${era.side > 0 ? "+" : "−"}${era.ride ? " ride" : ""}` +
        (era.unrealised_net != null ? ` · unreal ${(era.unrealised_net * 100).toFixed(2)}%` : "") +
        ` · eras ${era.totals?.eras ?? "—"} net ${era.totals?.net == null ? "—" : `${(era.totals.net * 100).toFixed(2)}%`}` +
        (era.totals?.ambiguous ? ` · ambiguous ${era.totals.ambiguous} (friction unresolved)` : "")
      : "era — waiting first window";
    const lock = c.last_lock && c.last_lock.outcome
      ? `last: ${c.last_lock.outcome} ${c.last_lock.bps != null ? (c.last_lock.bps >= 0 ? "+" : "") + c.last_lock.bps + "bp" : ""}`
      : "no sealed lock yet";
    return `<div class="hcell ${active ? "active" : ""}" data-cell="${key}" title="${esc(key)} — ${esc(c.rule || c.tier || "")}">
      <div class="hc-top"><span class="hc-sym">${sym} ${tf.toUpperCase()} · ${(a.tf_minutes || c.minutes)}m</span>
        <span class="hc-tier ${_tierCls(c.tier)}" title="${esc(c.tier_reason || "why this tier: not recorded")}">${esc(c.tier)}</span></div>
      ${c.tier_reason ? `<div class="hc-why dim" style="font-size:10px;line-height:1.3">${esc(c.tier_reason)}</div>` : ""}
      ${c.geometry_note ? `<div class="hc-rr" style="font-size:10px;line-height:1.3">${rrTxt
        ? `<b>${rrTxt}</b> ` : ""}${esc(c.geometry_note)}</div>` : ""}
      ${c.why ? `<div class="hc-why dim" style="font-size:10px;line-height:1.3">${esc(c.why)}</div>` : ""}
      <div class="hc-wr">${wr}${wrLabel} <span class="hc-conf">WR${conf ? " · " + conf : ""}${c.cost_infeasible
          ? ` · <span class="acc-dn">round trip ${num(c.round_trip_bps, 1)}bp &gt; the whole target</span>`
          : c.cannot_pay_at_any_target
            ? ` · <span class="hc-cost">COST: ${num(c.round_trip_bps, 1)}bp RT vs a `
              + `${num(c.reward_risk, 2)}x TP/SL → needs ${num((c.win_rate ?? 0) * 100 + 0, 0)}%· vs `
              + `${num(c.tp_multiple_of_rt_needed, 1)}x RT to pay</span>`
            : c.enforced_tp_bps != null
              ? ` · <span class="hc-cost">break-even target ${num(c.enforced_tp_bps, 0)}bp</span>`
              : c.round_trip_bps != null
                ? ` · <span class="hc-cost">round trip ${num(c.round_trip_bps, 1)}bp` + `${c.cost_note ? ` — ${esc(String(c.cost_note).slice(0, 80))}` : ""}</span>` : ""}</span></div>
      <div class="hc-side">${dir} · ${esc(c.rule || (c.certified ? "" : "no rule"))}</div>
      <div class="hc-era">${esc(eraTxt)}</div>
      <div class="hc-lock">${esc(lock)}</div>
      <div class="hc-timer">${locked
        ? `<span>⏱ ANALYSE locked</span><span class="cd" data-until="${untilMs}">⏳ ${_cdText(left) || "0:00"}</span>`
        : `<span>▶ ready</span><span class="free">free — click ANALYSE</span>`}</div>
    </div>`;
  }).join("");
  // per-card click = switch asset+tf (then the ANALYSE button targets it)
  box.querySelectorAll(".hcell").forEach(el => el.addEventListener("click", () => {
    const [sym, tf] = (el.dataset.cell || "").split("|");
    if (!sym) return;
    const ab = $("#seg-assets")?.querySelector(`[data-sym="${sym}"]`);
    if (ab) ab.click();
    const tb = $("#seg-tf")?.querySelector(`[data-tf="${tf}"]`);
    if (tb) tb.click();
    loadCells();
  }));
  // ANALYSE button state for the FOCUS cell: disabled while its timer runs
  const cur = cells[`${S.symbol}|${S.focusTf}`];
  const btn = $("#btn-analyse");
  if (btn && cur && !btn.dataset.busy) {
    const a = cur.analyse || {};
    const left2 = (a.locked_until || 0) * 1000 - now;
    if (left2 > 0) {
      btn.disabled = true;
      btn.title = `${S.symbol} ${S.focusTf.toUpperCase()} is inside its ${(a.tf_minutes || 22)}-minute analysis window — ${Math.ceil(left2 / 60000)} min left. Pick another cell or wait; timers are independent.`;
    } else {
      btn.disabled = false;
      btn.title = "Run the staged analysis for this asset+timeframe (arms this cell's timer only)";
    }
  }
}
/* 1 s countdown updater — runs inside THE one master setInterval */
function updateCellsTick() {
  if (!S.cells) return;
  const now = Date.now() + (S.serverOffset || 0);
  let expired = false;
  document.querySelectorAll(".hcell .cd[data-until]").forEach(el => {
    const left = Number(el.dataset.until) - now;
    if (left <= 0) { expired = true; return; }
    el.textContent = "⏳ " + _cdText(left);
  });
  if (expired && !S._cellsReload) {
    S._cellsReload = true;
    loadCells().finally(() => { S._cellsReload = false; });
  }
}

/* issue 3 — REST reconciliation for missed analyse events.
   Driven by THE one master setInterval (advice H-2) via a deadline flag —
   a second interval here would violate the single-timer contract. */
function startAnalysePoll() {
  S._anPollUntil = Date.now() + 180000;   // 3 min budget (watchdog is 175 s)
  S._anPollMisses = 0;
}
function stopAnalysePoll() { S._anPollUntil = 0; }
async function _analysePollStep() {
  if (S._anPollBusy) return;
  S._anPollBusy = true;
  try {
    const st = await api("/api/analyse/status");
    const job = st.job || {};
    if (job.status === "done" && job.result) {
      stopAnalysePoll();
      onAnalyseDone({ symbol: job.symbol, tf: job.tf, result: job.result });
    } else if (job.status === "error") {
      stopAnalysePoll();
      onAnalyseDone({ symbol: job.symbol, tf: job.tf, result: null,
                      error: job.error || "analysis failed" });
    } else if (job.status === "running") {
      S._anPollMisses = 0;
      (job.stages || []).forEach(onAnalyseStage);
    } else if (++S._anPollMisses > 15) {
      stopAnalysePoll();                       // idle elsewhere
    }
  } catch {
    if (++S._anPollMisses > 30) stopAnalysePoll();
  } finally { S._anPollBusy = false; }
}
async function resyncAnalyse() {
  try {
    const st = await api("/api/analyse/status");
    const job = st.job || {};
    if (job.status === "done" && job.result) {
      const panel = $("#analyse-panel");
      if (panel && panel.classList.contains("hidden")) return;  // not open
      onAnalyseDone({ symbol: job.symbol, tf: job.tf, result: job.result });
    } else if (job.status === "running") {
      const panel = $("#analyse-panel");
      if (panel) panel.classList.remove("hidden");
      $("#an-title").textContent = `ANALYSING ${job.symbol} · ${(job.tf || "tf1").toUpperCase()}`;
      if (!$("#an-stages")?.children.length) {
        $("#an-stages").innerHTML = (job.stages || []).map((st, i) => {
          const name = typeof st === "object" ? String(st?.stage || "?") : String(st);
          return `<div class="an-row" id="anr-${name}">
             <span class="dim">${i + 1}</span><span class="an-name">${name.replace("_", " ")}</span>
             <span class="an-ms">…</span><span class="an-at">—</span>
             <span class="an-sum dim">…</span></div>`; }).join("");
      }
      (job.stages || []).forEach(onAnalyseStage);
      startAnalysePoll();
    }
  } catch { /* offline */ }
}
function onPredictionSettled(msg) {
  // the engine sealed + settled locks on the tape — this used to be
  // emitted and IGNORED, so the book/WR rows stayed stale until a poll
  const locks = msg.locks || [];
  S.predStats = msg.stats || S.predStats || null;
  const won = locks.filter(l => l.outcome === "won").length;
  const lost = locks.filter(l => l.outcome === "lost").length;
  const ambiguous = locks.filter(l => l.outcome === "ambiguous").length;
  if (locks.length) {
    const last = locks[locks.length - 1];
    const tone = ambiguous && !won && !lost ? "" : won >= lost ? "ok" : "err";
    toast(`📊 updated ${locks.length} lock${locks.length > 1 ? "s" : ""} ` +
      `(+${won} / −${lost} / ?${ambiguous}) · ${last.symbol || ""} ${String(last.tf || "").toUpperCase()} ` +
      `${String(last.outcome || "").toUpperCase()}${last.bps != null ? " " + (last.bps >= 0 ? "+" : "") + Number(last.bps).toFixed(1) + "bp" : ""}`,
      tone);
  }
  if (S.view === "history") {
    loadSealedPredictions().catch(() => {});
    loadAnalyses().catch(() => {});
  }
  // rest of the app re-syncs from the next snapshot (single-timer rule)
}

function onAnalyseStarted(msg) {
  const panel = $("#analyse-panel");
  if (panel) panel.classList.remove("hidden");
  $("#an-title").textContent = `ANALYSING ${msg.symbol} · ${(msg.tf || "tf1").toUpperCase()}`;
  const started = msg.started_at ? new Date(msg.started_at * 1000) : new Date();
  $("#an-sub").textContent = `started ${started.toISOString().slice(11, 23)}Z — calculating…`;
  if (!$("#an-stages")?.children.length) {
    $("#an-stages").innerHTML = (msg.stages || []).map((name, i) =>
      `<div class="an-row" id="anr-${name}">
         <span class="dim">${i + 1}</span><span class="an-name">${String(name).replace("_", " ")}</span>
         <span class="an-ms">…</span><span class="an-at">—</span>
         <span class="an-sum dim">queued</span></div>`).join("");
  }
  startAnalysePoll();
}
function onAnalyseStage(msg) {
  // THREE shapes reach this handler and the old code only understood
  // the first: (1) the WS wrapper {stage: entry}; (2) the bare entry
  // itself — the resync/replay paths pass job.stages entries straight
  // through, so msg.stage was the STRING "formulas", st.stage was
  // undefined, and the row sat on "queued" forever ("it didn't use
  // formulas"); (3) a calculation_period entry {stage, at, ms}.
  let st = null;
  if (msg && typeof msg.stage === "object" && msg.stage) st = msg.stage;
  else if (msg && typeof msg.stage === "string") st = msg;
  if (!st || typeof st.stage !== "string") return;
  const row = $("#anr-" + st.stage);
  if (!row) return;
  row.className = "an-row " + (st.status || "done");
  const ms = st.duration_ms != null ? st.duration_ms : st.ms;
  const at = st.t_start_ms != null ? st.t_start_ms : st.at;
  if (ms != null) $(".an-ms", row).textContent = ms + " ms";
  if (at != null) $(".an-at", row).textContent = new Date(at).toISOString().slice(11, 23) + "Z";
  const sum = st.summary || "";
  $(".an-sum", row).textContent = sum;
  row.title = sum;
}
function onAnalyseDone(msg) {
  stopAnalysePoll();
  const b = $("#btn-analyse"); if (b) b.disabled = false;
  const r = msg.result;
  const sub = $("#an-sub");
  if (!r) {
    if (sub) sub.textContent = "FAILED — " + (msg.error || "unknown error");
    toast("analysis failed: " + (msg.error || ""), "err");
    return;
  }
  // issue 3: calculation_period may be absent on degraded results — an
  // unguarded deref threw here and aborted the whole render (the click
  // looked like it did nothing)
  const finAt = r.calculation_period && r.calculation_period.finished_at;
  if (sub) sub.textContent = finAt
    ? `finished ${new Date(finAt * 1000).toISOString().slice(11, 23)}Z · total ${r.elapsed_ms} ms`
    : `finished · total ${r.elapsed_ms || "—"} ms`;
  // HEAL rows whose live WS event was missed (reconnect, backgrounded
  // tab): the final calculation_period carries every stage's real
  // timing — a row still reading "queued" at done gets filled here,
  // never overwritten (only untouched rows are healed)
  (r.calculation_period?.stages || []).forEach(c => {
    const row = c && c.stage ? $("#anr-" + c.stage) : null;
    if (row && row.className.trim() === "an-row") onAnalyseStage({ stage: c });
  });
  const box = $("#an-result");
  if (!box) return;
  box.classList.remove("hidden");
  const dirCls = r.signal === "BUY" ? "buy" : r.signal === "SELL" ? "sell" : "flat";
  // REGIME DISAGREEMENT: the certified side STANDS — the big word is the
  // real side, and the disagreement is a banner beside it. (This used to
  // print "NO CALL" over a live, graded call: the contradiction was resolved
  // by deleting the answer.)
  const bigWord = r.signal;
  const conflictBanner = r.regime_conflict
    ? `<div class="conf acc-warn">⚠ REGIME DISAGREEMENT — the 5-state chip reads ${esc(r.regime_conflict.state)} (conf ${num(r.regime_conflict.regime_conf)}) while the certified side is ${esc(r.signal)}. The call stands and is graded; only ORDER ROUTING is held back until the tape agrees with itself. A withheld call is not a safer call — it is an ungraded one.</div>`
    : "";
  const ai = r.ai || {};
  // The review object now always carries available/backend/note, so the
  // panel can say WHICH brain answered and why it did not — "keyless" was
  // printed even when a local agent was loaded but silent.
  const aiWhere = ai.backend ? ` via ${esc(ai.backend)}` : "";
  let aiLine = ai.available
    ? Object.entries(ai.reviews || {}).map(([name, rv]) => {
        if (!rv || rv.ok === false) return `${name}: unavailable (${esc((rv || {}).error || "no reply")})`;
        const issues = (rv.errors || []).length;
        return `${name}${issues ? `: ${issues} issue(s) — ${esc(rv.reason || "")}` : ": concurs"}`;
      }).join(" · ") || "a reviewer answered but returned no readable verdict"
    : `${esc(ai.note || "no AI answered — deterministic pipeline only")}`;
  if (ai.available) aiLine = `${aiLine}${aiWhere}`;
  if (ai.correction) {
    aiLine += ` · ⚠ AI CORRECTION by ${esc(ai.correction.by)}: ${ai.correction.direction > 0 ? "BUY" : "SELL"} (${Math.round((ai.correction.confidence || 0) * 100)}%) — ${esc(ai.correction.reason || "")}`;
  }
  const period = (r.calculation_period?.stages || []).map(c =>
    `${c.stage} ${Math.round(c.ms)}ms`).join(" → ");
  box.innerHTML = `<div class="an-box">
    <div class="an-signal ${dirCls}">
      <div class="big">${bigWord}</div>
      ${conflictBanner}
      <div class="conf">confidence ${fx(r.confidence, 3)} · ${r.verdict === "ISSUE" ? "ISSUED" : (r.verdict || (r.gate_ok ? "WATCH" : "NO CALL"))}${r.issue_note ? ` · ${esc(r.issue_note)}` : ""}${r.cell_certainty != null ? ` · certainty ${fx(r.cell_certainty, 3)} (n=${r.cell_certainty_n ?? "?"})` : ""}</div>
      ${r.cell_win_rate_net != null ? `<div class="conf">real win rate ${fx(r.cell_win_rate_net * 100, 1)}% after ${fx(r.cell_round_trip_bps ?? 0, 1)} bps round trip · gross ${fx((r.cell_win_rate ?? 0) * 100, 1)}%</div>`
        : (r.cell_win_rate != null ? `<div class="conf acc-warn">gross hit rate ${fx(r.cell_win_rate * 100, 1)}% — no post-cost win rate is recorded for this cell${r.cell_min_tp_bps != null ? `; a win only pays if the target clears ${fx(r.cell_min_tp_bps, 1)} bps (${fx(r.cell_round_trip_bps ?? 0, 1)} bps round trip × 1.5)` : ""}</div>` : "")}
      ${r.prediction_lock?.locked ? `<div class="conf acc-up">🔒 LOCKED ${esc(r.prediction_lock.side)} · ${esc(String(r.prediction_lock.id || "").slice(-10))} · TP ${px(r.prediction_lock.tp)} / SL ${px(r.prediction_lock.sl)} · saved to history</div><div class="hint">Prediction tracking only — not connected to external demo fills. Outcome is graded from completed market bars; same-bar TP/SL is marked ambiguous, never guessed.</div>` : `<div class="conf acc-warn">lock refused: ${esc((r.prediction_lock?.why || ["no finite ladder"]).join("; "))}</div>`}
      ${r.price_action ? `<div class="conf">📈 price action · ${r.price_action.direction > 0 ? "bullish" : r.price_action.direction < 0 ? "bearish" : "neutral"} · score ${fmt(r.price_action.score, 2)} · ${esc(r.price_action.reason || "no contextual setup")} · no confidence boost</div>` : ""}
      <div class="conf">entry ${px(r.entry ?? r.price)} · ATR ${fmt(r.atr, 4)} · TP:SL ${r.ratio ? Number(r.ratio).toFixed(2) + ":1" : "—"} · ${esc(r.levels_source || "preview only")}</div>
      ${(() => { const pol = r.prediction_lock?.evidence?.hold_policy; if (!pol) return "";
        const wr = pol.win_rate != null ? `${(pol.win_rate * 100).toFixed(1)}%` : "—";
        const lo = pol.confidence != null ? ` · wilson lo ${(pol.confidence * 100).toFixed(1)}%` : "";
        const hd = pol.era_ev_net != null ? ` · era EV ${(pol.era_ev_net * 100).toFixed(2)}%/era net after ${(pol.round_trip_friction_bps || 24)}bp RT friction` : "";
        const cf = pol.cost_floor;
        const cfLine = cf ? `<div class="conf ${cf.tp_widened_for_cost ? "acc-warn" : "acc-up"}">💰 cost floor: target ${fx(cf.tp_bps_certified, 1)} bps vs ${fx(cf.min_tp_bps, 1)} bps required (1.5× ${fx(cf.rt_bps, 1)} bps round trip) — ${esc(cf.note)}</div>` : "";
        return `<div class="conf acc-up">🎯 HOLD POLICY [${esc(pol.tier || "?")}] ${esc(pol.rule || pol.side || "")} · certified GROSS hit rate ${wr}${lo}${hd}</div>` + cfLine +
          (pol.scale_source ? `<div class="conf">row geometry k=${pol.row_k} m=${pol.row_m} · ${esc(String(pol.scale_source).slice(0, 90))}</div>` : ""); })()}
      ${(() => { const ev = r.prediction_lock?.evidence; if (!ev) return "";
        const fs = Object.entries(ev.formula_signs || {}).slice(0, 8)
          .map(([k, v]) => `F${k}${Number(v) > 0 ? "+" : "−"}`).join(" ");
        const bs = (ev.brain_signals || []).slice(0, 8);
        return `<div class="conf">🧮 formulas used: ${fs || "—"} · 🧠 brain signals: ${bs.length ? bs.map(x => Number(x) > 0 ? "▲" : Number(x) < 0 ? "▼" : "●").join("") : "—"}</div>`; })()}
      ${r.committee ? `<div class="conf">🧠 committee: ${esc(r.committee.effect || "—")}${r.committee.agree_share != null ? ` · agree ${(r.committee.agree_share * 100).toFixed(0)}%` : ""}${r.committee.consensus != null ? ` · consensus ${r.committee.consensus >= 0 ? "+" : ""}${Number(r.committee.consensus).toFixed(2)}` : ""} · conf ${fx(r.committee.confidence_before, 3)}→${fx(r.committee.confidence_after, 3)}${r.committee.direction_held ? " · side held (certified)" : ""}${(r.committee.vetoes || []).length ? ` · ⚠ ${esc(r.committee.vetoes[0])}` : ""}</div>` : ""}
      ${(r.committee.votes || []).length ? `<div class="conf">votes: ${r.committee.votes.map(v => `${esc(v.agent)} <b>${v.direction > 0.05 ? "▲" : v.direction < -0.05 ? "▼" : "●"}</b> ${Math.round(v.confidence * 100)}%${v.veto ? " ⚠" : ""} <i>${esc(String(v.reason || "").slice(0, 64))}</i>`).join(" · ")}</div>` : ""}
      ${r.metacog ? (() => { const mc = r.metacog; const nst = Object.keys(mc.statuses || {}).length;
        const sym = msg.symbol || S.symbol;
        const mine = Object.entries(mc.statuses || {}).filter(([k]) => k.startsWith(sym + "|"));
        const w = Object.entries(mc.weights || {}).find(([k]) => k.startsWith(sym + "|"));
        const stTxt = mine.map(([k, v]) => `${k.split("|")[1].toUpperCase()} <b>${esc(String(v))}</b>`).join(" · ");
        return `<div class="conf">🧠 online meta-cognition: ${nst} cells live${stTxt ? " · " + stTxt : ""}${w ? ` · exp3 w ${esc(JSON.stringify(w[1]).slice(0, 60))}` : ""}</div>`; })() : ""}
      ${(r.signal === "BUY" || r.signal === "SELL") ? chopChip(msg.tf || S.focusTf) : ""}
      <div class="conf">${r.contract_state === "locked" ? `LOCKED CONTRACT ${esc((r.contract_id || "").slice(0, 16))}` : "PREVIEW ONLY — NO LOCKED CONTRACT"} · ${r.execution_eligible ? "EXECUTION ELIGIBLE" : (r.recheck?.passed ? "PAPER ONLY — signal produced, orders not routed (tape: " + esc((r.tape_mode || (S.snap?.feed?.source) || "sim")) + ")" : "NOT EXECUTION-ELIGIBLE")}</div>
    </div>
    <div class="an-grid">
      ${r.contract_tp ? `<div class="an-cell"><div class="k">LOCKED CONTRACT TP</div><div class="v" style="color:#26a69a">${px(r.contract_tp)}</div></div>` : ""}
      ${r.tps.map(t => `<div class="an-cell"><div class="k">MEASURED RUNG ${t.k} · ${t.minutes}m</div><div class="v">${px(t.level)}</div></div>`).join("")}
      <div class="an-cell"><div class="k">LOCKED CONTRACT SL</div><div class="v" style="color:#f0385a">${r.contract_sl ? px(r.contract_sl) : r.sl ? px(r.sl) : "—"}</div></div>
      <div class="an-cell"><div class="k">LADDER</div><div class="v" style="font-size:11.5px;font-weight:400">${esc(r.ladder?.source || "—")}</div></div>
      <div class="an-w">weightage for ${msg.symbol} (per-asset mix): ${(r.weightage || []).map(w => `F${w.id} <b>${w.w}</b>`).join(" · ")}<br/><span class="dim">${esc(r.weightage_note || "")}</span></div>
      <div class="an-ai">AI review — ${esc(aiLine)}</div>
      ${r.recheck ? `<div class="an-recheck" style="margin-top:6px;font-size:11.5px">
        <b class="${r.recheck.passed ? "acc-up" : "acc-dn"}">${esc(r.recheck.verdict)}</b>
        ${r.recheck.score != null ? `<div class="dim">weighted opinion score
          <b>${fmt(r.recheck.score, 2)}</b> / needed ${fmt(r.recheck.threshold, 2)}
          ${r.recheck.hard_failed && r.recheck.hard_failed.length ? ` · data blockers: ${esc(r.recheck.hard_failed.join(", "))}` : " · data checks passed"}
          ${(r.recheck.failed || []).length ? ` · soft flags: ${esc((r.recheck.failed || []).join(", "))}` : ""}</div>` : ""}
        ${(r.recheck.checks || []).map(c => `<div class="kv"><span>${c.ok ? "✓" : "✗"} ${esc(c.name)} ${c.hard ? "<i>(hard)</i>" : c.weight ? `<i>w ${fmt(c.weight, 2)}</i>` : ""}</span><b class="${c.ok ? "" : "acc-dn"}">${esc(c.detail)}</b></div>`).join("")}
        ${r.recheck.voters ? `<div class="kv"><span>voter signs</span><b>${Object.entries(r.recheck.voters).map(([k, v]) =>
          `<span class="${v > 0 ? "acc-up" : v < 0 ? "acc-dn" : "dim"}">${esc(k)} ${v > 0 ? "+1" : v < 0 ? "−1" : "0"}</span>`).join(" · ")}</b></div>` : ""}
        ${r.prediction ? `<div class="hint">${r.prediction.kind === "locked-contract" ? "locked contract" : "preview only"}: ${r.prediction.direction > 0 ? "BUY" : r.prediction.direction < 0 ? "SELL" : "FLAT"} · entry ${px(r.prediction.entry)} · TP ${px(r.prediction.tp)} · SL ${px(r.prediction.sl)} · TP:SL ${r.prediction.ratio != null ? Number(r.prediction.ratio).toFixed(2) + ":1" : "—"} · ${r.prediction.execution_eligible ? "execution eligible" : "paper only — signal stands, no order is routed on this tape"}</div>` : `<div class="hint">raw analysis read was ${esc(r.raw_signal || "—")} — not produced as a prediction (see failed checks above).</div>`}
      </div>` : ""}
      <div class="an-period">calculation period: ${esc(period)} · total ${r.elapsed_ms} ms</div>
    </div></div>`;
  // SYNC: the analysis recomputes every subsystem — refetch whatever view
  // is open so formulas/emotion/physics/brain/news all show the same run
  refreshActiveView(true);
  loadCells();                             // the cell's timer is armed now
  toast("analysis complete — all views synced", "ok");
}
function renderRail() {
  const a = S.snap?.assets || {};
  ASSETS.forEach(sym => {
    const d = a[sym]; if (!d) return;
    const pr = S.prices[sym] ?? d.price;
    const el = $("#ap-" + sym); if (el) el.textContent = px(pr);
    const ch = $("#ac-" + sym);
    if (ch) {
      ch.textContent = `${d.window_change_pct >= 0 ? "+" : ""}${fmt(d.window_change_pct, 2)}%`;
      ch.className = "chg " + (d.window_change_pct >= 0 ? "up" : "down");
    }
  });
  const g = S.snap?.grc || {};
  $("#rs-grc").textContent = g.count ?? "—";
  $("#rs-next").textContent = g.seconds_to_next != null
    ? `${Math.floor(g.seconds_to_next / 60)}m ${Math.floor(g.seconds_to_next % 60)}s` : "—";
  $("#rs-phase").textContent = (S.snap?.phase || "—").toUpperCase();
  // The engine publishes mode "binance" | "sim" (see app/feed.py) — it has
  // NEVER published "live". Comparing against "live" made a real Binance
  // tape print "SIM (regime-switching)" in the rail, and the status chip
  // below (chips.feed === "live") made the dashboard chip read "SIM FEED"
  // on a live tape. Both now compare against "binance" and fall back to the
  // snapshot's own `feed.simulated` flag when the per-asset mode is absent.
  const _feedMode = a[BTC_MODE()]?.mode || S.snap?.feed?.mode || "unknown";
  const _sim = (typeof S.snap?.feed?.simulated === "boolean")
    ? S.snap.feed.simulated : _feedMode !== "binance";
  $("#feed-mode").textContent = _sim
    ? `SIM (regime-switching)${S.snap?.feed?.real_error ? " — exchange unreachable" : ""}`
    : "LIVE (Binance)";
  const led = (id, on, warn) => {
    const e = $(id); if (e) e.className = "led " + (warn ? "warn" : on ? "on" : "off");
  };
  led("#led-stream", S.lastMsgAt && Date.now() - S.lastMsgAt < 5000);
  led("#led-learn", true, S.snap?.learning?.high_alert);
  // brain connection — same component the brain tab shows, mirrored here
  // so the dashboard can never say "nothing" while the brain is green
  try {
    // the LIVE component state (snapshot.brain) is the one source of truth;
    // the per-(asset,tf) copy is rebuilt at GRC and used to paint "brain
    // offline" over a healthy connectome on a not-yet-re-analysed window
    const lb = S.snap?.brain || null;
    const fly = S.snap?.systems?.[`${S.symbol || "BTC"}|${S.focusTf || "tf1"}`]?.brain?.flywire;
    let bc = lb?.chip
      || fly?.chip
      || (fly?.ready ? { kind: "ok", text: "FlyWire v783 · live" }
        : { kind: "absent", text: "brain — " + (fly?.reason || "no state yet") });
    // "state": the component's own machine. A chip that only knows
    // ok/build/absent cannot tell "loading" from "broken", which is exactly
    // how a downloading connectome ended up painted as "brain offline —
    // vote withheld (RECONNECT)". An unready brain whose state is not a
    // failure is BLUE (working), never red.
    const bstate = lb?.state || fly?.state || null;
    if (bc.kind !== "ok" && (bstate === "building" ||
        (bstate && bstate !== "ready" && bstate !== "absent"))) {
      bc = { kind: "build",
             text: lb?.chip?.text || fly?.chip?.text ||
                   ("brain — loading (" + bstate + ")") };
    }
    if (bstate === "building" && lb?.chip) {
      bc = { kind: "build", text: lb.chip.text, stalled: !!lb.chip.stalled };
    }
    // A READY CONNECTOME WITH NO VOTE YET IS NOT OFFLINE. `brain.vote` is null
    // until the first GRC pass produces one, and `systems[].brain` is only
    // rebuilt at GRC — so on a freshly booted board every window reported
    // "brain offline — vote withheld" while FlyWire was ready with 138k
    // neurons (the screenshot's red chip over a healthy brain tab). If the
    // component says ready, the chip says ready and explains the pending vote.
    if (lb?.ready && (!bc || bc.kind !== "ok")) {
      bc = { kind: "ok", text: `FlyWire ready · vote pending until the next GRC`
             + (lb.pass_age_s != null ? ` (pass ${Math.round(lb.pass_age_s)}s ago)` : "") };
    }
    const snapAge = S.snap?.ts ? (Date.now() / 1000 - S.snap.ts) : null;
    if (snapAge != null && snapAge > 20 && bc.kind !== "ok") {
      bc = { kind: "build", text: `brain — state stale ${Math.round(snapAge)}s (snapshot lagging)` };
    }
    // a REBUILD is not an outage: hold the last green for 90 s, then be honest
    if (bc.kind === "ok") S._brainOkAt = Date.now();
    else if (bc.kind === "absent" && S._brainOkAt && Date.now() - S._brainOkAt < 90000) {
      bc = { kind: "build", text: "brain — re-verifying (was ready " +
             Math.round((Date.now() - S._brainOkAt) / 1000) + "s ago)" };
    }
    led("#led-brain", bc.kind === "ok", bc.kind === "build");
    if (bc.kind === "absent") led("#led-brain", false, false);
    const bchip = $("#brain-chip");
    if (bchip) {
      bchip.textContent = "🧠 " + bc.text;
      // a build that has gone silent is not "working": gold = progressing,
      // orange = the supervisor is retrying, red = a real, reported failure
      const cc = bc.kind === "ok" ? "var(--green)"
        : bc.kind === "build" ? (bc.stalled ? "var(--orange)" : "var(--gold)")
        : "var(--red)";
      bchip.style.color = cc;
      bchip.style.borderColor = cc;
      bchip.classList.add("live");
    }
  } catch (e) { /* chip optional */ }
  led("#led-agent", (S.snap?.agents?.slots || []).some(s => s.status === "connected"));
  const chip = $("#mode-chip");
  const mode = S.snap?.agents?.mode || "drosophila";
  chip.textContent = mode === "drosophila" ? "DROSOPHILA" : mode === "local" ? "LOCAL MODEL" : "CLOUD";
  chip.className = "mode-" + mode;
}
function BTC_MODE() { return S.symbol; }

/* ═══════════ WHY panel ═══════════ */
function renderWhy() {
  const el = $("#why-panel"); if (!el) return;
  const tf = S.focusTf;
  const s = S.snap?.systems?.[`${S.symbol}|${tf}`];
  if (!s) { el.innerHTML = `<div class="dim pad">warming up…</div>`; return; }
  const bd = s.breakdown || {}, m = bd.multipliers || {}, votes = bd.votes || {};
  // a regime conflict withholds the side here as well (see renderHero)
  const _cf = s.regime_conflict || null;
  const dir = _cf ? 0 : (s.direction ?? 0);
  const comps = (bd.components || []).slice(0, 10);
  const syn = s.synthesis;
  const synBlock = syn
    ? (syn.ok
      ? `<div class="kv-row"><div class="kv"><span>committee regime</span><b>${syn.regime}</b></div>
         <div class="kv"><span>confidence</span><b>${fmt(syn.confidence_score, 2)}</b></div></div>
         <div class="hint">Groq ${syn.model || ""} · ${syn.rationale || ""}</div>`
      : `<div class="hint">committee: ${syn.reason || "no key"}</div>`)
    : "";
  const lk = s.lock || {};
  const lockAge = lk.locked_at ? Math.max(0, Math.round((Date.now() + (S.serverOffset || 0)) / 1000 - lk.locked_at)) : null;
  const lockBlock = lk.locked_at ? `
    <div class="lock-block ${lk.relocks ? "relocked" : ""}">
      <b>🔒 LOCKED ${new Date(lk.locked_at * 1000).toISOString().slice(11, 19)}Z</b>
      ${lk.side > 0 ? "▲ BUY" : lk.side < 0 ? "▼ SELL" : "● NO SIDE"}
      · entry ${px(lk.price)} · conf ${fmt(lk.confidence, 3)}
      ${lk.emotion ? `· emotion <i>${lk.emotion}</i>` : ""}
      ${lk.relocks ? `· ⚡ emergency re-lock ${new Date((lk.relock_at || 0) * 1000).toISOString().slice(11, 19)}Z` : ""}
      <div class="dim">locked reasoning (immutable until next window): ${String(lk.reason || "").slice(0, 220)}</div>
      <div class="dim">live values below refresh continuously — they do NOT change the lock.</div>
    </div>` : "";
  // PRICE PROVENANCE — the user's "chart price and analyse price don't
  // match": they are different instants of the same tape, not two markets.
  // Print each one with its own timestamp and the drift between them.
  const pa = S.snap?.assets?.[S.symbol] || {};
  const chp = pa.chart || {};
  const hhmmss = (ts) => ts ? new Date(ts * 1000).toISOString().slice(11, 19) + "Z" : "—";
  const nowPx = (S.prices?.[S.symbol] > 0) ? S.prices[S.symbol] : pa.price;
  const entryPx = lk.price ?? s.entry ?? null;
  const driftBps = (entryPx > 0 && nowPx > 0) ? ((nowPx / entryPx) - 1) * 1e4 : null;
  const priceBlock = (nowPx > 0) ? `
    <div class="lock-block">
      <b>💲 PRICE PROVENANCE</b>
      now ${px(nowPx)} <span class="dim">@ ${hhmmss(pa.price_asof)} · tick ${pa.feed_age_s ?? "?"}s old</span>
      ${entryPx > 0 ? ` · entry ${px(entryPx)} <span class="dim">locked @ ${hhmmss(lk.locked_at || s.cycle_start)}</span>` : ""}
      ${driftBps != null ? ` · drift <b class="${Math.abs(driftBps) > 25 ? "down" : ""}">${driftBps >= 0 ? "+" : ""}${driftBps.toFixed(1)} bp</b> since the lock` : ""}
      ${(() => { const w = S.snap?.assets?.[S.symbol]?.tfs?.[S.tf] || {};
        if (w.relocks == null) return "";
        return ` · window ${fmt(w.entry_age_s, 0)}s old · ` + (w.relocks > 0
          ? `<b class="down">⚡ ${w.relocks} re-lock(s)</b> at ${hhmmss(w.relocked_at)}`
          : "no re-lock (anchor stands)"); })()}
      ${chp.last_bar_close ? `<div class="dim">chart candle ${px(chp.last_bar_close)} @ ${hhmmss(chp.last_bar_t)} (${chp.age_s ?? "?"}s ago) · ${chp.delta_bps >= 0 ? "+" : ""}${fmt(chp.delta_bps, 1)} bp vs spot — same tape, one instant behind; the lock is the anchor, not spot</div>` : ""}
    </div>` : "";
  // advice §8.1/8.2: WHICH ladder rung decided the side + weighted mass
  const lad = s.ladder || null;
  const mass = s.mass || null;
  const ladderBlock = lad ? `
    <div class="lock-block">
      <b>🧭 DIRECTION LADDER</b> ${esc(lad.source || "")}
      <div class="dim">noise floor ${fmt(lad.noise_floor, 4)} · fused score ${fmt(lad.score, 5)}</div>
      ${mass ? `<div class="dim">weighted mass for <b class="up">${fmt(mass.for, 4)}</b>
        vs against <b class="down">${fmt(mass.against, 4)}</b>
        · top for: ${(mass.supporters || []).map(x => `${esc(x.name)} ${fmt(x.value, 2)}`).join(", ") || "—"}
        · top against: ${(mass.opponents || []).map(x => `${esc(x.name)} ${fmt(x.value, 2)}`).join(", ") || "—"}</div>
      <div class="dim">subsystem votes (never vote — modulate only):
        ${Object.entries(mass.subsystem_votes || {}).map(([k, v]) => `${esc(k)} ${fmt(v, 4)}`).join(" · ") || "—"}</div>` : ""}
    </div>` : "";
  // confidence trend sparkline (last readings, floor0.35 marked by scale)
  if (!S.confHist) S.confHist = [];
  const confNow = Number(s.confidence || 0);
  const lastC = S.confHist[S.confHist.length - 1];
  if (!lastC || Math.abs(lastC.v - confNow) > 1e-9) S.confHist.push({ v: confNow });
  S.confHist = S.confHist.slice(-40);
  const chv = S.confHist.map(pp => pp.v);
  const cmin = Math.min(0, ...chv), cmax = Math.max(0.35, ...chv);
  const pts = chv.map((vv, i) =>
    `${(i / Math.max(1, chv.length - 1)) * 100},${28 - ((vv - cmin) / (cmax - cmin || 1)) * 26}`).join(" ");
  const sparkHtml = `<div style="text-align:right"><svg viewBox="0 0 100 30" preserveAspectRatio="none" style="width:130px;height:26px;vertical-align:middle"><polyline points="${pts}" fill="none" stroke="${confNow > 0.35 ? "#22c55e" : "#38bdf8"}" stroke-width="1.5"/></svg><span class="dim" style="margin-left:6px">confidence trend</span></div>`;
  // contribution ticks for this window
  const tk = s.ticks || {};
  const cnt = (g) => g ? Object.values(g).filter(x => x.on).length + "/" + Object.values(g).length : "—";
  const offF = (Object.entries(tk.formulas || {})
    .filter(([, x]) => !x.on).slice(0, 6)
    .map(([k, x]) => `${x.name || "F" + k}`).join(" · ")) || "none";
  const ticksBlock = Object.keys(tk).length ? `
    <div class="lock-block"><b>✓ CONTRIBUTION TICKS — this window</b>
      <div class="votes" style="margin-top:4px">
        <span class="vote ${cnt(tk.formulas).startsWith("0/") ? "" : "pos"}" title="formulas with a non-zero contribution to CCI">formulas ${cnt(tk.formulas)}</span>
        <span class="vote ${cnt(tk.emotions).startsWith("0/") ? "" : "pos"}" title="emotion channels above 0.01">emotions ${cnt(tk.emotions)}</span>
        <span class="vote ${cnt(tk.physics).startsWith("0/") ? "" : "pos"}" title="physics mechanisms with coverage">physics ${cnt(tk.physics)}</span>
        <span class="vote ${cnt(tk.hedge).startsWith("0/") ? "" : "pos"}" title="hedge legs non-zero">hedge ${cnt(tk.hedge)}</span>
        <span class="vote ${tk.news?.aggregate?.on ? "pos" : ""}" title="news aggregate this window">news ${tk.news?.aggregate?.on ? "✓ on" : "○ off"}</span>
        <span class="vote ${tk.brain?.flywire?.on ? "pos" : ""}" title="FlyWire connectome vote">brain ${tk.brain?.flywire?.ready ? (tk.brain.flywire.on ? "✓ on" : "○ flat") : "absent"}</span>
      </div>
      <div class="dim">formulas ✗ this window: ${esc(offF)} — per-item ticks live in FORMULAS / EMOTION / PHYSICS.</div>
    </div>` : "";
  const exBlock = `<div style="margin-top:8px"><button class="btn" id="why-explain-btn" onclick="window.toggleWhyExplain()" style="padding:4px 12px">🧠 explain this + TP/SL math</button><pre id="why-explain" class="logic" style="white-space:pre-wrap;display:none;margin-top:6px"></pre></div>`;
  // WHO DECIDED — subsystem share of |contribution| (old-app inspiration)
  const who = bd.who || {};
  const whoTotal = Object.values(who).reduce((x2, y) => x2 + (Number(y) || 0), 0);
  const whoBlock = whoTotal > 0 ? `
    <div class="lock-block"><b>WHO DECIDED — share of |contribution|</b>
      <div class="votes" style="margin-top:4px">
        ${Object.entries(who).map(([k, v]) =>
          `<span class="vote ${(v || 0) > 0.05 ? "pos" : ""}">${esc(k)} ${Math.round((v || 0) * 100)}%</span>`).join("")}
      </div>
      <div class="dim">the money-weight share each subsystem had in the fused score this window (subsystem votes still only modulate — never set — the direction).</div>
    </div>` : "";
  // CONVICTION — edge vs noise floor + size advice (old-app inspiration)
  const _score = Number(lad?.score ?? bd.sum_contributions ?? 0);
  const _nf = Number(lad?.noise_floor ?? 0.02);
  const _edge = Math.abs(_score) - _nf;
  const _conf = Number(s.confidence || 0);
  const _neutral = _edge <= 0;
  const _size = !s.issued
    ? "call: side pending first compute"
    : (_neutral || _conf < 0.5)
      ? "size: half or less — thin edge, keep the stop tight"
      : "size: normal — edge and confidence both speak";
  const convBlock = `
    <div class="lock-block" style="border-color:var(--gold)">
      ⚠ CONVICTION — ${_neutral ? "inside the neutral band" : "decisive edge"} ·
      edge <b>${fx(_edge)}</b> over noise floor ${fx(_nf)} ·
      confidence <b>${fx(_conf)}</b>
      <div class="dim">${_size} · fused score ${fx(_score)} · window ${esc(String(s.ladder?.source || "—").slice(0, 60))}</div>
    </div>`;
  el.innerHTML = lockBlock + priceBlock + ladderBlock + sparkHtml + ticksBlock + whoBlock + convBlock + exBlock + `
    <div class="why-head">
      <div class="why-dir ${dir > 0 ? "up" : dir < 0 ? "down" : "flat"}">
        ${dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO CALL"}<span class="sym">${S.symbol} · ${tf.toUpperCase()} · ${TF_META[tf].mins}m</span>${_cf ? ` <span class="acc-warn" title="${esc(_cf.state)} (conf ${num(_cf.regime_conf)}) opposes this side — the call stands and is graded, no order is routed">· ${esc(_cf.state)} opposes · no routing</span>` : ""}</div>
      <div class="why-conf">CONFIDENCE
        <div class="bar"><i class="${s.confidence > 0.35 ? "pos" : "neg"}" style="width:${clamp((s.confidence || 0) * 100, 0, 100)}%"></i></div>
        <b>${fx(s.confidence, 3)}</b> / floor ${bd.floor ?? 0.35}</div>
      <div class="why-badge ${s.issued && !_cf ? "ok" : ""}">${_cf
        ? `⚠ REGIME DISAGREEMENT · ${esc(_cf.state)} vs the certified side — call stands and is graded, no order routed`
        : s.issued
        ? `✓ SIGNAL ${dir > 0 ? "BUY" : dir < 0 ? "SELL" : "—"} · ${esc(s.tier || "watch-only")} · P ${fmt(s.p_side, 2)} score-units${s.tier === "watch-only" ? ` · size 0${s.tier_why ? " — " + esc(s.tier_why) : " (below measured noise floor)"}` : ""}`
          + (s.calib_n != null ? `<span class="dim" title="${esc(s.calib_source || "")}"> · n=${s.calib_n}${s.calib_provisional ? " (provisional scale)" : ""}</span>` : "")
        : "side pending"}</div>
    </div>
    <div class="why-reason">${s.reason || ""}</div>
    <div class="why-grid">
      <div>
        <h5>TOP CONTRIBUTORS — ω × F(adj) · ${bd.n_components || "?"} components total</h5>
        ${comps.map(c => `
          <div class="contrib">
            <span class="cname" title="${c.name}">${c.name}</span>
            <div class="bar"><i class="${c.contribution >= 0 ? "pos" : "neg"}"
              style="width:${clamp(Math.abs(c.contribution) * 320, 2, 100)}%"></i></div>
            <span class="val">${c.contribution >= 0 ? "+" : ""}${fmt(c.contribution, 4)}</span>
            <span class="wv">ω${fmt(c.weight, 3)} × ${c.value >= 0 ? "+" : ""}${fmt(c.value, 2)}</span>
          </div>`).join("")}
        <div class="hint">Σ contributions = <b>${fmt(bd.sum_contributions, 4)}</b> = CCI. ω weights relearn every GRC (reward = sign agreement with the move).</div>
      </div>
      <div>
        ${(() => {
          // the 1:1 floor is a hard rule; when it moved a level the card
          // must say so, or the published ratio looks like it fell out of
          // the maths instead of being corrected by policy.
          const fl = s.rr_floor && s.rr_floor.before;
          if (!fl || fl >= 1) return "";
          return `<div class="kv"><span class="warn">1:1 FLOOR</span>`
            + `<b class="dim">target raised from ${fmt(fl, 3)}:1 to 1:1 `
            + `(stop unchanged) — a trade may never pay less than it risks`
            + `</b></div>`;
        })()}
        <h5>CONFIDENCE CHAIN — spec order</h5>
        ${(() => {
          // WHY THE NUMBER IS WHAT IT IS (user report 2026-11-27: "In
          // explain this why confidence is too low you can do anything to
          // increase confidence"). The engine now records every stage that
          // MOVED the published confidence (`conf_trace`). Print it, and
          // name the binding constraint: a chain that ends at the 0.02
          // floor because |CCI| is inside its own noise floor is a real
          // "no edge", not a broken formula — and the reader deserves to
          // see which of the two it is.
          const t = s.conf_trace;
          if (!Array.isArray(t) || !t.length) return "";
          const floorHit = (s.confidence ?? 0) <= 0.0205;
          const snr = s.snr;
          const head = floorHit
            ? `<div class="kv"><span class="acc-warn">confidence is at the floor</span>`
              + `<b class="dim">the fused score sits inside this cell's own noise floor`
              + (snr != null ? ` (snr ${fmt(snr, 2)})` : "")
              + `, so the calibrated base is ~0.02 — the chain is working, `
              + `there is no edge to report in this window. It rises when `
              + `|CCI_ADJ| clears the floor, and the trace below is the only `
              + `place that can say otherwise.</b></div>`
            : "";
          return head
            + `<div class="kv"><span class="dim">published</span><b>${fmt(s.confidence, 4)}`
            + ` <span class="dim">· floor ${fmt(s.asset_floor, 3)}</span></b></div>`
            + t.map(x => `<div class="kv"><span>${esc(x.stage)}</span>`
              + `<b class="dim">${x.before == null ? "—" : fmt(x.before, 4)} → `
              + `${fmt(x.after, 4)}${x.why ? ` <span class="dim">(${esc(x.why)})</span>` : ""}</b></div>`).join("");
        })()}
        ${(() => {
          // The walk-forward edge model's own probability. It is the only
          // directional component in this system with a measured
          // out-of-sample edge at the published reward:risk (39.9% at 2:1,
          // CI 38.8-41.1, vs 33.3% break-even), so its number belongs on the
          // card beside the side it moved — not buried in a log.
          const em = s.edge_model;
          if (!em) return "";
          const up = em.prob_up >= 0.5;
          const col = up ? "acc-up" : "acc-down";
          return `<div class="kv"><span>EDGE MODEL</span>`
            + `<b class="${col}">P(up) ${fmt(em.prob_up, 3)} · `
            + `${up ? "UP" : "DOWN"} · ±${fmt(em.confidence, 3)}</b></div>`
            + `<div class="kv"><span class="dim" title="measured out-of-sample, `
            + `walk-forward with purge; break-even at 2:1 is 33.3%">`
            + `horizon ${esc(em.horizon_bars)}h · stop ${esc(em.stop_atr)}×ATR · `
            + `${esc(em.source || "")}</span></div>`;
        })()}
        ${(() => {
          // The chain is now regenerated on the CCI that ACTUALLY SHIPPED
          // (it used to describe the pre-fusion score while the card printed
          // a rescaled one). When even that final pass is moved by a later
          // stage, `chain_gap` says so instead of letting two numbers sit
          // side by side unexplained.
          const cg = s.chain_gap;
          if (!cg) return "";
          return `<div class="kv"><span class="acc-warn" title="the chain below is regenerated on the published CCI; these later stages (fusion / asset behaviour / advanced layer) moved the final number">⚠ chain total ${fmt(cg.chain, 3)} vs published ${fmt(cg.published, 3)}</span><b class="dim">${esc(cg.why || "")}</b></div>`;
        })()}
        <div class="kv"><span>1 · |CCI_adj| × min(1, DRHE ${fmt(m.drhe, 2)})</span><b>${fx(bd.base_confidence, 4)}</b></div>
        <div class="kv"><span>2 · × (0.7 + 0.3 × EQ ${fmt(m.eq, 2)})</span><b>brain B2</b></div>
        <div class="kv"><span>3 · state "${m.dominant || "—"}" ${m.state_aligned ? "aligned → +" + fmt(0.1 * (m.p_dominant || 0), 3) : "not aligned"}</span><b>${m.state_aligned ? "aligned" : "—"}</b></div>
        <div class="kv"><span>4 · RTP ${fmt(m.rtp, 2)}${m.rtp > 0.6 ? " ⚠" : ""} · PCV ${fx(m.pcv, 3)}${m.pcv < 0 ? " ⚠" : ""} ${(s.breakdown || {}).penalty_capped ? `<span class="acc-warn" title="penalties exceeded the available confidence — residue kept, never an exact 0">capped</span>` : ""}</span><b>penalties</b></div>
        <div class="kv"><span>5-7 · ×(0.5+0.5·WFCP ${fmt(m.wfcp, 2)}) ×(1−DRPT ${fmt(m.drpt, 2)}) ×(1−GSPI ${fmt(m.gspi, 2)})</span><b>${fx(bd.final_confidence, 4)}</b></div>
        <h5 style="margin-top:12px">SUBSYSTEM VOTES</h5>
        <div class="votes">${Object.entries(votes).map(([k, v]) =>
          `<span class="vote ${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${k} ${v > 0 ? "▲" : v < 0 ? "▼" : "·"}</span>`).join("")}</div>
        ${(() => {
          const f = s.brain?.flywire;
          if (!f) return "";
          const tip = (f.top_channels || []).map(c => `F${c.fid} ${c.value >= 0 ? "+" : ""}${fmt(c.value, 3)}`).join(" · ");
          return f.ready
            ? `<div class="kv" style="margin-top:6px"><span title="${esc(tip)}">brain ${f.vote >= 0 ? "+" : ""}${fmt(f.vote, 3)} <span class="dim">(FlyWire v783 · conf ${fmt(f.conf, 2)}${tip ? " · hover for top channels" : ""})</span></span><b class="${f.vote >= 0 ? "up" : "down"}">${f.vote >= 0 ? "approach (leans in · bullish)" : "avoid (leans away · bearish)"}</b></div>`
            : `<div class="kv" style="margin-top:6px"><span>brain</span><b class="down" title="${esc(f.reason || "")}">absent (${esc(f.reason || "vote withheld")}) — weight 0</b></div>`;
        })()}
        <div class="hint">4-vote double-check: PCV + agent + brain (+ emotion/physics here) must agree to issue.</div>
        ${(() => {
          const a = s.advanced; if (!a) return "";
          const mv = Object.entries(a.modules || {}).map(([k, v]) =>
            `<span class="vote ${v.vote > 0 ? "pos" : v.vote < 0 ? "neg" : ""}" title="${k} vote ${fmt(v.vote, 3)}${v.regime ? " · regime " + v.regime : ""}${v.hurst != null ? " · H=" + fmt(v.hurst, 2) : ""}${v.z != null ? " · z=" + fmt(v.z, 2) : ""}${v.why ? " · not voting: " + v.why : ""}">${k}</span>`).join("");
          const cf = a.conformal;
          return `<h5 style="margin-top:12px">ADVANCED MATH LAYER</h5>
          <div class="kv-row">
            <div class="kv"><span>advanced vote (blended in)</span><b class="${a.vote > 0 ? "up" : a.vote < 0 ? "down" : ""}">${a.vote > 0 ? "+" : ""}${fmt(a.vote, 3)}</b></div>
            <div class="kv"><span>soft gate × (entropy·OU·BOCPD)</span><b>${fmt(a.soft_gate, 2)}</b></div>
            <div class="kv"><span>RQA confidence × <span class="dim" title="Raw RQA determinism is ~0.44 even on pure noise (small-sample diagonal bias), so the multiplier is measured AGAINST that null: only determinism above it counts. A multiplier pinned at a bound would be a constant, not a signal.">det ${fmt(a.rqa_det ?? s.advanced?.rqa_det, 2)} vs null ${fmt(a.rqa_null ?? s.advanced?.rqa_null, 2)}</span></span><b>${fmt(a.rqa_mult, 3)}</b></div>
            <div class="kv"><span>size × (HAR·EVT·RMT)</span><b>${fmt(a.size_mult, 2)}</b></div>
            ${a.rmt_mode_share != null ? `<div class="kv"><span>market-mode share</span><b>${fmt(a.rmt_mode_share, 2)}</b></div>` : ""}
            ${a.evt?.q99_bps ? `<div class="kv"><span>EVT 1-in-100 move</span><b>${fmt(a.evt.q99_bps, 0)} bps</b></div>` : ""}
            ${cf && cf.q_bps != null ? `<div class="kv"><span>conformal 80% band</span><b>±${fmt(cf.q_bps, 0)} bps (n=${cf.n})</b></div>` : cf ? `<div class="kv"><span>conformal 80% band</span><b class="dim">${esc(cf.state || "calibrating")} (n=${cf.n || 0})</b></div>` : ""}
          </div>
          <div class="votes">${mv}</div>
          <div class="hint">kalman · fracdiff · bocpd · entropy · ou/hurst · jumps · pairs-ou · wavelet · signature(floor) · rmt-residual — every module votes; soft gates only scale confidence/size, never veto a side.</div>`;
        })()}
      </div>
    </div>
    ${synBlock ? `<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--line2)"><h5>GROQ COMMITTEE SYNTHESIS</h5>${synBlock}</div>` : ""}`;
}

/* ═══════════ SETTLEMENT LEDGER (advice G-1: one accuracy source) ═══════ */
function renderLedger() {
  const el = $("#ledger-panel"); if (!el) return;
  const led = S.snap?.ledger; if (!led) return;
  const rows = Object.entries(led.per || {}).map(([key, st]) => {
    const [sym, tf] = key.split("|");
    const rateTxt = st.rate == null ? "—"
      : (st.enough ? `${Math.round(st.rate * 100)}%` : `${Math.round(st.rate * 100)}% <span class="dim">(n&lt;${led.min_n}: not enough windows)</span>`);
    return `<tr class="${sym === S.symbol ? "focus-row" : ""}">
      <td class="mono">${sym}</td><td class="mono">${tf.toUpperCase()}</td>
      <td class="mono" title="${st.seeded ? st.seeded + " of these settled rows are 🌱 seeded (24h real-tape replay)" : "all rows settled live"}">${st.n}${st.seeded ? ` <span class="dim">🌱${st.seeded}</span>` : ""}</td>
      <td class="mono acc-up">${st.wins}</td><td class="mono acc-dn">${st.losses}</td>
      <td class="mono">${st.skipped}</td>
      <td class="mono">${rateTxt}</td>
      <td class="mono" title="mean signed bps per graded window · payoff = avg win ÷ avg loss (bps)">${st.exp_bps != null ? `${st.exp_bps >= 0 ? "+" : ""}${fmt(st.exp_bps, 1)}bp` : "—"}${st.payoff != null ? ` · ×${fmt(st.payoff, 2)}` : ""}</td>
      <td class="mono">${st.streak} / ${st.best_streak}</td>
      <td class="mono">${(st.last20 || []).map(r => r === "win" ? "W" : "L").join("") || "·"}</td>
    </tr>`;
  }).join("");
  const pooled = Object.entries(led.pooled || {}).map(([tf, st]) =>
    `<div class="kv"><span>${tf.toUpperCase()} pooled (all assets)</span><b>${st.n} settled · ${st.rate != null ? Math.round(st.rate * 100) + "%" : "—"}${st.enough ? "" : " · n&lt;" + led.min_n}</b></div>`).join("");
  // settlement liveness — a wall of zeros must explain ITSELF
  const ll = $("#led-live");
  if (ll) {
    if (led.last_settle_ago_s != null) {
      ll.textContent =
        `${led.settle_calls} settles · last ${led.last_settle_ago_s}s ago` +
        ` (${led.last_settle?.tf || ""} ${led.last_settle?.result || ""})` +
        ` · totals ${led.totals?.win ?? 0}W/${led.totals?.loss ?? 0}L/${led.totals?.skip ?? 0}skip` +
        ` · rows ${led.rows_total ?? 0}`;
    } else {
      ll.textContent =
        "no window settled yet — windows are epoch-aligned: tf1 fills at :00/:22/:44, tf2 at :00/:44 (needs n≥20 to speak a rate)";
    }
  }
  el.innerHTML = `<table class="tbl">
    <thead><tr><th>ASSET</th><th>TF</th><th>n</th><th>W</th><th>L</th><th>SKIP</th><th>STRICT-SIGN</th><th>EXPECTANCY</th><th>STREAK/BEST</th><th>LAST 20</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="kv-row" style="margin-top:8px">${pooled}</div>
    <div class="hint">${led.rule} — every accuracy number in the app is read from this ledger, nowhere else.</div>`;
}

/* ═══════ HINDSIGHT (advice §8.3): z-graded multipliers, printed ═══════ */
function renderHindsight() {
  const el = $("#hindsight-panel"); if (!el) return;
  const h = S.snap?.hindsight; if (!h) { el.innerHTML = `<div class="dim">no re-grade data yet</div>`; return;
  }
  const per = Object.values(h.per || {});
  const enough = per.filter(e => e.enough);
  const inverted = enough.filter(e => e.inverted);
  const damped = [...enough].sort((a, b) => a.multiplier - b.multiplier).slice(0, 8);
  const changes = (h.changes || []).slice().reverse().slice(0, 6);
  el.innerHTML = `
    <div class="kv-row">
      <div class="kv"><span>graded (n≥${h.min_decided})</span><b>${enough.length}</b></div>
      <div class="kv"><span>not enough windows</span><b>${per.length - enough.length}</b></div>
      <div class="kv"><span>inverted (z≤−3, n≥16)</span><b>${inverted.length}</b></div>
      <div class="kv"><span>cadence</span><b>${Object.entries(h.cadence || {}).map(([k, v]) => k.toUpperCase() + " " + v).join(" · ")}</b></div>
    </div>
    <table class="tbl" style="margin-top:6px">
      <thead><tr><th>SUBJECT</th><th>n</th><th>HITS</th><th>MISSES</th><th>RATE</th><th>z</th><th>MULT</th><th>NOTE</th></tr></thead>
      <tbody>${damped.map(e => `<tr class="${e.inverted ? "acc-dn" : ""}">
        <td>${esc(e.label)}</td><td>${e.n}</td><td class="acc-up">${e.hits}</td>
        <td class="acc-dn">${e.misses}</td><td>${Math.round(e.rate * 100)}%</td>
        <td>${e.z}</td><td><b>${e.multiplier}</b></td>
        <td class="dim">${esc(e.changes_note || "")}</td></tr>`).join("")}</tbody>
    </table>
    <div class="hint">Changes applied at the window boundary: ${changes.map(c =>
      `<span class="chip chip-warn">${esc(c.formula)} ${c.from}→${c.to} (${esc(c.reason)})</span>`).join(" ") || "none yet"}</div>
    <div class="hint">${esc(h.rule || "")}</div>`;
}

/* ═════ META (§8.4) + PREDICTABILITY GAUGE (§8.5) — honest answers ═════ */
function renderMeta() {
  const el = $("#meta-panel"); if (!el) return;
  const m = S.snap?.meta; if (!m) { el.innerHTML = `<div class="dim">no meta data yet</div>`; return; }
  const ev = m.evidence || {};
  const rec = m.record || {};
  const gauge = m.predictability || null;
  const bits = gauge && gauge.ready ? gauge.bits : null;
  el.innerHTML = `
    <div class="kv-row">
      <div class="kv"><span>settled rows</span><b>${m.n}</b></div>
      <div class="kv"><span>40-window record</span><b>${rec.rate != null ? Math.round(rec.rate * 100) + "%" : "—"}${rec.takeover ? " · ✓ TAKEOVER" : ""}</b></div>
      <div class="kv"><span>predictability</span><b title="${bits != null && bits < 0.001 ? "0.000 bit = no measurable signal in the window yet (H₂ at or above 1 bit); it rises when outcomes become predictable" : ""}">${bits != null ? bits.toFixed(3) + " bit" : (gauge?.note || "needs 40 windows")}</b></div>
      <div class="kv"><span>calib a · b</span><b>${ev.calib_a ?? "—"} · ${ev.b ?? "—"}</b></div>
    </div>
    <div class="hint">${esc(rec.message || "")} · ${esc(m.expert?.message || "")}</div>
    ${gauge && gauge.ready ? `<div class="hint ${gauge.bits < 0.02 ? "chip-warn" : "chip-ok"}">P(up)=${gauge.p_up} · H₂=${gauge.h2_bits} bit — ${esc(gauge.note)}</div>` : ""}
    <table class="tbl" style="margin-top:6px">
      <thead><tr><th>SOURCE</th><th>slope aᵢ</th><th>MI (bits)</th><th>coefficient</th></tr></thead>
      <tbody>${Object.entries(ev.slopes || {}).map(([k, v]) => {
        const mi = (ev.mi_bits || {})[k];
        return `<tr>
        <td>${esc(k)}</td><td>${v}</td><td>${mi ?? "—"}${mi != null && Number(mi) === 0 ? ' <span class="dim" title="no mutual information measured for this source in the current window">no signal</span>' : ""}</td>
        <td>${(m.coefficients || {})[k] ?? "—"}</td></tr>`; }).join("")}</tbody>
    </table>
    <div class="hint">${esc(m.rule || "")}</div>`;
}

/* ═══════════ CALLS & TARGETS — TP/SL for every timeframe ═══════════ */
function renderCalls() {
  const el = $("#calls-panel"); if (!el) return;
  const a = S.snap?.assets?.[S.symbol] || {};
  $("#calls-scope").textContent = `${S.symbol} · every main timeframe (sub-windows are TF targets, no separate toggle)`;
  // TF-FOCUS: once the user picks a timeframe the calls table shows ONLY
  // that window (all-TF mode until then, toggle in the hero)
  const rows = TF.filter(tf => !S.tfFocus || tf === S.focusTf).map(tf => {
    const cyc = a.tfs?.[tf];
    const res = S.snap?.systems?.[`${S.symbol}|${tf}`];
    const m = TF_META[tf];
    const dir = cyc?.direction ?? res?.direction ?? 0;
    const _weak = String(res?.ladder?.source || "").includes("inside noise floor")
      || Number(res?.confidence || 0) < 0.05;
    const dirTxt = dir > 0
      ? `<b class="${_weak ? "flat" : "up"}">▲ BULL${_weak ? "*" : ""}</b>`
      : dir < 0
        ? `<b class="${_weak ? "flat" : "down"}">▼ BEAR${_weak ? "*" : ""}</b>`
        : '<b class="flat">● FLAT</b>';
    const _leanNote = _weak
      ? `<br/><span class="dim" style="font-size:11px">* weak lean inside noise floor — never issued</span>` : "";
    // TP4 = THIS timeframe's own horizon → ALWAYS shown/calc-first
    const _tg = cyc?.targets || [];
    const _ord = _tg.map((_, i) => i);
    if (_tg.length >= 4) _ord.unshift(_ord.pop());   // [3,0,1,2]
    const tps = _ord.map(i => {
      const t = _tg[i];
      if (!t) return "";
      const st = t.status === "hit" ? '<i class="st hit">HIT</i>'
        : t.status === "locked" ? '<i class="st lock">LOCK</i>' : '<i class="st open">open</i>';
      const isTf = i === _tg.length - 1;
      const lbl = isTf ? `TF ${m.mins}m` : `TP${i + 1} · ${m.tps[i]}m`;
      return `<td class="tp-cell">${t.level ? px(t.level) : "—"}<br/><span class="tp-min${isTf ? " tp-horizon" : ""}">${lbl}</span> ${st}</td>`;
    }).join("");
    const sl = cyc?.sl ? `${px(cyc.sl)}${cyc.sl_provisional ? ' <span class="dim" style="font-size:11px">prov</span>' : ""}` : "—";
    return `<tr class="${tf === S.focusTf ? "focus-row" : ""}">
      <td class="mono">${tf.toUpperCase()}</td>
      <td class="mono">${m.mins}m</td>
      <td>${dirTxt} <span class="dim">· ${verdictOf(cyc, res)}${cyc?.tier ? " · " + cyc.tier : ""}</span>${_leanNote}</td>
      <td class="mono">${fmt(res?.confidence, 3)}</td>
      ${tps}
      <td class="mono sl-cell">${sl}</td>
      <td class="mono">${cyc?.ratio ? Number(cyc.ratio).toFixed(2) + ":1" : "—"}</td>
    </tr>`;
  }).join("");
  el.innerHTML = `<table class="tbl calls-tbl">
    <thead><tr><th>TF</th><th>WINDOW</th><th>CALL</th><th>CONF</th>
      <th>TF-TP (horizon)</th><th>TP1</th><th>TP2</th><th>TP3</th><th>SL</th><th>RATIO</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="hint">The TF's own horizon TP (the window itself) is always calculated first and shown first; TP1–TP3 are its sub-timeframes (one stop-loss, one ratio). Levels come from the engine's ATR projection at every GRC.${S.tfFocus ? " Focused: only " + S.focusTf.toUpperCase() + " shown — use ⊞ all timeframes." : ""}</div>`;
}

/* ═══════════ candles (per symbol, per TF) ═══════════ */
function candleRevisionFallback(bars) {
  // Older servers may not send a revision. Hash completed OHLC content so
  // equal counts/timestamps cannot make the chart retain a stale tape.
  let h = 2166136261;
  for (const b of (bars || []).slice(0, -1)) {
    const row = `${b.t}|${b.o}|${b.h}|${b.l}|${b.c}|${b.v || 0};`;
    for (let i = 0; i < row.length; i++)
      h = Math.imul(h ^ row.charCodeAt(i), 16777619);
  }
  return (h >>> 0).toString(16);
}
async function loadCandles() {
  const symbol = S.symbol;
  const seq = (S._candleLoadSeq || 0) + 1;
  S._candleLoadSeq = seq;
  // issue 4: the two fetches are INDEPENDENT. They used to share one
  // try block, so a failed /api/candles (busy server, slow network)
  // skipped /api/tf-charts entirely and every mini-chart stayed blank.
  // §6 7-10-day warmup: ask for the whole backfill window (server cap
  // = BACKFILL_BARS, 7 days) so boot renders real history immediately
  // instead of WARMING_UP / n=0 dead states
  const mdP = api(`/api/candles?symbol=${symbol}&tf=tf1&n=10080`)
    .catch(() => null);
  // guide §1: the SERVER publishes tf_charts[asset][tf] with one step and
  // one lookback per timeframe. Cards bind to that series when it is
  // available and fall back to local aggregation only if it is not, so
  // two cards of the same TF can no longer disagree.
  const tfcP = api(`/api/tf-charts?symbol=${symbol}`)
    .catch(() => null);
  try {
    let [md, tfc] = await Promise.all([mdP, tfcP]);
    if (!md && !tfc) return;                       // both failed — keep state
    // Ignore an older response after a symbol switch or a newer refetch;
    // otherwise BTC bars can be written into another asset's chart state.
    if (seq !== S._candleLoadSeq || symbol !== S.symbol) return;
    if (md && (md.symbol && md.symbol !== symbol || !Array.isArray(md.bars)))
      return;
    // md may be null when only tf-charts answered — degrade to an empty
    // minute series so the server-painted cards below still render
    md = md || { bars: [] };
    md.bars ||= [];
    const mbars = (md.bars || []).map(x =>
      ({ time: x.t, open: x.o, high: x.h, low: x.l, close: x.c, v: x.v || 0 }));
    // EACH timeframe card aggregates the tape at its OWN interval
    // (user: every TF chart must be different — they were all the same
    // 1-minute series photocopied four times). Wider TF = wider bars.
    const QINT = { tf1: 1, tf2: 1, tf3: 5, tf4: 5 };   // doc C-2 resolutions
    const QWIN = { tf1: 22, tf2: 44, tf3: 130, tf4: 247 };  // own horizon each
    if (S.tfBarsSymbol !== symbol) {
      S.tfBars = {}; S.tfBarsRevision = {}; S.tfBarsSymbol = symbol;
    }
    S.tfBars ||= {}; S.tfBarsRevision ||= {};
    S.tfBarsForeign = Math.max(0, Number(md.excluded_foreign_bars) || 0);
    S.tfBarsIntegrity = (md.integrity && typeof md.integrity === "object")
      ? md.integrity : null;
    S.tfBarsFetchedAt = Date.now();
    S.tfBarsFamily = String(md.bar_family || "unknown");
    S.tfBarsMode = String(md.feed?.mode || "unknown");
    const historyRevision = String(md.revision ||
      `client-${candleRevisionFallback(md.bars)}`);
    const _cut = (bars, tf) => {
      const intervalMin = QINT[tf] || 1;
      const windowMin = QWIN[tf] || 60;
      // Three completed horizon windows plus the current one, with a
      // hard floor of 30 real candles. Preserve the chart spec's 90-bar
      // first-paint context for the 22-minute / 1m view.
      const lookbackMin = Math.max(4 * windowMin, 30 * intervalMin,
        tf === "tf1" ? 90 * intervalMin : 0);
      const tmax = bars.length ? bars[bars.length - 1].t : 0;
      return bars.filter(b => b.t >= tmax - lookbackMin * 60);
    };
    const agg = (bars, mins) => {
      if (mins <= 1) return bars;
      const out = new Map();
      for (const b of bars) {
        const t = Math.floor(b.t / (mins * 60)) * mins * 60;
        const o = out.get(t);
        if (!o) out.set(t, { t, o: b.o, h: b.h, l: b.l, c: b.c, v: b.v || 0 });
        else { o.h = Math.max(o.h, b.h); o.l = Math.min(o.l, b.l); o.c = b.c; o.v += b.v || 0; }
      }
      return [...out.values()];
    };
    await Promise.all(TF.map(async tf => {
      const srv = (tfc && tfc.ok && tfc.charts) ? tfc.charts[tf] : null;
      // a short server series (or a failed /api/tf-charts) must never
      // blank the card: fall back to local aggregation instead
      const useSrv = !!(srv && Array.isArray(srv.candles)
                        && srv.candles.length >= 20);
      const bars = useSrv ? srv.candles : agg(md.bars || [], QINT[tf] || 1);
      const mapped = useSrv
        ? srv.candles.map(x => ({ time: x.t, open: x.o, high: x.h,
                                  low: x.l, close: x.c, v: x.v || 0,
                                  ...(x.live ? { live: true } : {}) }))
        : _cut(bars, tf).map(x =>
            ({ time: x.t, open: x.o, high: x.h, low: x.l, close: x.c, v: x.v || 0 }));
      S.tfBars[tf] = mapped;
      // each TF keeps its OWN revision and epoch — a tf4 card must not
      // re-render because tf1 got a new minute
      S.tfBarsRevision[tf] = useSrv ? String(srv.revision || historyRevision)
                                    : historyRevision;
      S.tfEpoch ||= {};
      if (useSrv) S.tfEpoch[tf] = srv.candles[0].t;
      const c = S.charts[tf]; if (!c) return;
      if (!mapped.length) {
        c.setData([]);
        if (tf === S.focusTf) {
          S.priceHist = []; S.candleSeries = { symbol, bars: [] };
        }
        return;
      }
      const markers = [];
      for (let i = 2; i < bars.length - 2; i++) {
        const b = bars[i];
        if (b.h > bars[i-1].h && b.h > bars[i-2].h && b.h > bars[i+1].h && b.h > bars[i+2].h)
          markers.push({ time: b.t, above: true, color: "#f0385a", glyph: "◆" });
        if (b.l < bars[i-1].l && b.l < bars[i-2].l && b.l < bars[i+1].l && b.l < bars[i+2].l)
          markers.push({ time: b.t, above: false, color: "#00d4f0", glyph: "◆" });
      }
      const cyc = S.snap?.assets?.[symbol]?.tfs?.[tf];
      (cyc?.targets || []).forEach(t => {
        if (t.status === "hit" && t.hit_ts)
          markers.push({ time: Math.floor(t.hit_ts / 60) * 60, above: true, color: "#22c98a", glyph: "▲" });
      });
      c.setData(mapped);
      // seed the realised-path history (hero line chart + branches):
      // without this the white path only existed from page load and
      // the forward fan floated with nothing behind it
      if (tf === S.focusTf) {
        const _norm = t => t > 1e12 ? t / 1000 : t;
        const seeded = mapped.map(x => ({ t: _norm(x.time), p: x.close, v: x.v || 0 }))
          .filter(p2 => p2.p > 0);
        const tail = (S.priceHist || [])
          .filter(p2 => p2.t > (seeded[seeded.length - 1]?.t || 0));
        S.priceHist = [...seeded, ...tail].slice(-900);
        // full-window candle series for the charts' realised line:
        // covers the window from its start (REST backfill), so the
        // line begins at the window's left edge on a true time axis
        S.candleSeries = { symbol, bars: seeded };
      }
      // Binance-mode cards refresh on the same cadence (chart <-> price link)
      if (S.chartSrc === "binance") { try { bnEnsure(); } catch (e) {} }
    }));
    if (seq === S._candleLoadSeq && symbol === S.symbol) {
      try { drawMainChart(); } catch { /* chart may still be mounting */ }
    }
  } catch { /* first GRC may not be ready */ }
}

/* ═══════════ FORMULA LAB ═══════════ */
async function loadFormulas() {
  try {
    const [fh, sys] = await Promise.all([
      api(`/api/formula-history?symbol=${S.symbol}&tf=${S.focusTf}`),
      api(`/api/systems?symbol=${S.symbol}`),
    ]);
    const s = sys.systems?.[`${S.symbol}|${S.focusTf}`] || {};
    const fv = S.snap?.formula_values?.[`${S.symbol}|${S.focusTf}`] || {};
    const bd = s.breakdown || {};
    $("#fl-scope").textContent = `${S.symbol} · ${S.focusTf.toUpperCase()} · ${TF_META[S.focusTf].mins}m`;
    $("#fl-summary").innerHTML = `
      <div class="kv"><span>CCI raw</span><b>${fmt(s.cci, 4)}</b></div>
      <div class="kv"><span>CCI adjusted</span><b>${fmt(s.cci_adjusted, 4)}</b></div>
      <div class="kv"><span>base confidence</span><b>${fmt(bd.base_confidence, 4)}</b></div>
      <div class="kv"><span>final confidence</span><b>${fmt(s.confidence, 4)} ${s.confidence > 0.35 ? "✓" : ""}</b></div>
      <div class="kv"><span>direction</span><b>${s.direction > 0 ? "▲ BULL" : s.direction < 0 ? "▼ BEAR" : "● FLAT"}</b></div>
      <div class="kv"><span>verdict</span><b>${verdictOf(s, bd)}</b></div>`;
    const contribs = {};
    (bd.components || []).forEach(c => contribs[c.id] = c);
    const cards = $("#fl-cards"); cards.innerHTML = "";
    const core = fv.raw || {};
    // advice §2.1-2.4: per-formula trace (µs, span, guard, zero reason,
    // double check) from the engine's last traced pass
    let tr = {};
    try {
      tr = (await api(`/api/formula-traces?symbol=${S.symbol}&tf=${S.focusTf}`)).traces || {};
    } catch { tr = {}; }
    Object.keys(core).sort((a, b) => a - b).forEach(fid => {
      const raw = core[fid], adj = (fv.adj || {})[fid] ?? raw;
      const name = (fv.names || {})[fid] || `F${fid}`;
      const w = S.snap?.weights?.[fid] ?? contribs[fid]?.weight ?? 0;
      const contrib = contribs[fid]?.contribution;
      const hist = fh.core?.[fid] || [];
      const mn = hist.length ? Math.min(...hist) : null, mx = hist.length ? Math.max(...hist) : null;
      const mean = hist.length ? hist.reduce((a, b) => a + b, 0) / hist.length : null;
      const card = document.createElement("div");
      card.className = "card";
      const ft = tr[fid] || null;
      const hz = (S.snap?.hindsight?.per || {})[`${S.symbol}|${S.focusTf}|F${fid}`];
      // a 0.00 must ALWAYS carry its reason on the card itself (user:
      // "formulas 1 emotion is 0.00 without any reason") — trace reason
      // first, flat-history derivation as the guaranteed fallback
      const _flatWhy = (raw === 0 && hist.length > 2
        && hist.every(v => Math.abs(v - hist[0]) < 1e-12))
        ? `reading identical across the last ${hist.length} passes — the tape feature this formula reads did not change (flat input, not a dead formula)` : "";
      const zeroChip = (ft?.zero_reason || _flatWhy)
        ? `<span class="chip chip-warn" title="why it is 0">${ft?.zero_cause ? (ZCAUSE[ft.zero_cause] || ft.zero_cause) + " · " : ""}0 · ${ft?.zero_reason || _flatWhy}</span>` : "";
      const tickOn = contribs[fid] ? Math.abs(contribs[fid].contribution) >= 1e-5 : null;
      const tickChip = tickOn === null ? "" :
        `<span class="chip ${tickOn ? "chip-ok" : "chip-err"}" title="${tickOn ? "contributed to this window's CCI" : "did not contribute (flat or zero this window)"}">${tickOn ? "✓ tick" : "✗ no tick"}</span>`;
      const dc = ft?.double_check;
      const dcChip = dc
        ? `<span class="chip ${dc.ok ? "chip-ok" : "chip-err"}" title="${esc(dc.identity || "")}">${dc.label}</span>` : "";
      const traceLine = (ft || hz)
        ? `<span class="dim">${ft ? `${ft.us} µs · span ${ft.span}/${ft.guard_min ?? "–"} bars` : ""}${hz ? ` · z ${hz.z ?? "—"} · ${Math.round((hz.rate ?? 0) * 100)}% (${hz.hits ?? "?"}/${hz.n ?? "?"})` : ""}${ft?.non_directional ? " · non-directional (never votes)" : ""}</span>`
        : "";
      card.innerHTML = `<h4>${tickChip} <span>F${fid} · ${name}</span><span>${contrib >= 0 ? "+" : ""}${contrib != null ? fmt(contrib, 4) : "—"}</span></h4>
        <div class="kv-row">
          <div class="kv"><span>raw</span><b>${raw >= 0 ? "+" : ""}${fmtSig(raw)}</b></div>
          <div class="kv"><span>Bgn-adj</span><b>${adj >= 0 ? "+" : ""}${fmtSig(adj)}</b></div>
          <div class="kv"><span>weight ω</span><b>${fmt(w, 4)}</b></div>
        </div>
        <div class="kv-row trace-row">${zeroChip}${dcChip}${traceLine}</div>
        <div class="hint">how it came: raw <b>${fx(raw)}</b> → bgn <b>${fx(adj)}</b> → ω <b>${fx(w, 3)}</b> → contribution <b>${contrib != null ? (contrib >= 0 ? "+" : "") + fx(contrib, 4) : "—"}</b></div>
        ${hist.length > 2 ? `<canvas class="spark" height="46" data-f="${fid}"></canvas>
        <div class="kv-row"><div class="kv"><span>min/mean/max (100 GRC)</span><b>${fmt(mn, 3)} / ${fmt(mean, 3)} / ${fmt(mx, 3)}</b></div></div>` : ""}
        <div class="logic">${dc?.identity ? esc(dc.identity) + " · " : ""}${FORMULA_LOGIC[fid] || ""}</div>`;
      cards.appendChild(card);
    });
    // extras F28 ARC / F30 PCV
    if (fv.extra) {
      const ex = document.createElement("div");
      ex.className = "card";
      ex.innerHTML = `<h4><span>F28/F30 validators</span></h4>
        <div class="kv-row">
          <div class="kv"><span>ARC · conf/vol band</span><b>${fmt(fv.extra["28"], 4)}</b></div>
          <div class="kv"><span>PCV · conflict</span><b>${fmt(fv.extra["30"], 4)}</b></div>
        </div>
        <div class="logic">${FORMULA_LOGIC[28]} · ${FORMULA_LOGIC[30]}</div>`;
      cards.appendChild(ex);
    }
    // advice §2.7/§2.9: flat watch + forward P(up) superposition (4 columns)
    try {
      const fwd = await api(`/api/forward?symbol=${S.symbol}`);
      const flat = S.snap?.flat_watch?.[`${S.symbol}|${S.focusTf}`] || [];
      const flatEl = $("#fl-flat");
      if (flatEl) {
        flatEl.innerHTML = flat.length
          ? flat.map(r => `<span class="chip chip-warn" title="${esc(r.zero_reason || "flat reading")}"
              >F${r.fid} ${Math.round(r.share * 100)}% flat · n=${r.n}${r.zero_reason ? " · " + esc(r.zero_reason) : ""}</span>`).join("")
          : `<span class="dim">no formula stuck flat in the last 50 locks</span>`;
      }
      const fwEl = $("#fl-forward");
      if (fwEl) {
        const rows = Object.keys(core).sort((a, b) => a - b).map(fid => {
          // Every cell shows a real number (audit §6.1): Laplace-smoothed
          // bucket posterior when this reading's bucket has samples, else
          // the pooled per-(formula, asset, tf) posterior. A "°" marks a
          // thin sample (n < 30) — the number is shown, never implied to
          // be settled.
          const tds = ["tf1", "tf2", "tf3", "tf4"].map(tf => {
            const est = fwdEstimate(fwd.buckets?.[`F${fid} ${S.symbol} ${tf}`] || [],
                                    Math.max(-1, Math.min(1, core[fid])));
            if (!est) {
              return `<td class="dim" title="no settled window in this bucket yet">—</td>`;
            }
            const note = est.exact ? `bucket ${est.bucket}, n=${est.n}`
              : est.smoothed
                ? `smoothed from ±${est.span} pooled reading, n=${est.n}`
                : `pooled ±${est.span} reading span, n=${est.n}`;
            return `<td title="${note}" class="${est.exact ? "" : "fwd-pooled"}">${(est.p * 100).toFixed(0)}%`
              + `<span class="dim"> n${est.n}${est.exact ? "" : est.smoothed ? " · thin" : " · ±" + est.span}</span></td>`;
          }).join("");
          const cur = (fv.raw?.[fid] ?? core[fid]);
          const tOn = contribs[fid] ? Math.abs(contribs[fid].contribution) >= 1e-5 : null;
          const tk = tOn === null ? "–" : tOn ? "✓" : "✗";
          return `<tr><td title="${tOn ? "contributed this window" : tOn === null ? "no reading" : "did not contribute"}">${tk} F${fid} ${(fv.names?.[fid] || "")}</td><td>${fx(cur, 3)}</td>${tds}</tr>`;
        }).join("");
        fwEl.innerHTML = `<table class="fwd-table"><thead><tr><th>formula ✓/✗</th><th>now</th>
          <th>tf1 P(up)</th><th>tf2 P(up)</th><th>tf3 P(up)</th><th>tf4 P(up)</th></tr></thead>
          <tbody>${rows}</tbody></table>
          <div class="hint">Forward P(up) is a walked-forward frequency from this ledger. n≥30 in the exact reading bucket → plain %; below that the reading is POOLED with neighbouring buckets (±0.5 then ±1.0) and marked, and if the whole table is still thin the cell shows a Laplace-smoothed (up+1)/(n+2) frequency tagged <b>thin</b> — never an unexplained dash. Current value beside it: superposition (§2.8).</div>`;
      }
    } catch (e) { console.warn("forward/flat:", e); }
    // §7 genesis: pool counts, next generation, top formula + 7 metrics
    try {
      const [gs, gf] = await Promise.all([
        api("/api/genesis/status"), api("/api/genesis/formulas"),
      ]);
      const ge = $("#fl-genesis");
      if (ge) {
        const st = gs.states || {};
        const rows = (gf.formulas || []).slice(0, 6);
        ge.innerHTML = `<div class="kv-row">
          <div class="kv"><span>pool</span><b>${gs.pool}</b></div>
          <div class="kv"><span>states</span><b>${Object.entries(st).map(([k, v]) => `${k} ${v}`).join(" · ") || "empty"}</b></div>
          <div class="kv"><span>next generation</span><b>in ${gs.next_generation_in_s}s</b></div>
          <div class="kv"><span>passes</span><b>${gs.generation_count}</b></div>
        </div>
        <table class="tbl" style="margin-top:6px">
          <thead><tr><th>ID</th><th>DOMAIN</th><th>STATE</th><th>HIT</th><th>IC</th><th>STAB</th><th>TURN</th><th>max|ρ|</th><th>µs</th><th>n</th></tr></thead>
          <tbody>${rows.map(f => `<tr class="${f.state === "retired" ? "acc-dn" : ""}">
            <td title="${esc(f.definition)}">${esc(f.id.split(":")[1] || f.id)}</td>
            <td>${esc(f.domain_name)}</td><td>${f.state}</td>
            <td>${f.metrics.hit_rate ?? "—"}</td><td>${f.metrics.ic ?? "—"}</td>
            <td>${f.metrics.stability ?? "—"}</td><td>${f.metrics.turnover ?? "—"}</td>
            <td>${f.metrics.max_corr ?? "—"}</td><td>${f.metrics.latency_us ?? "—"}</td>
            <td>${f.metrics.n ?? 0}</td></tr>
            ${f.autopsy ? `<tr><td colspan="10" class="dim">🩺 ${esc(f.autopsy)}</td></tr>` : ""}`).join("")}</tbody>
        </table>
        <div class="hint">${esc(gs.rules || "")}</div>`;
      }
    } catch (e) {
      // issue 7: never leave the panel on "loading…" — show the error
      // and a retry link instead (console.warn alone was invisible)
      console.warn("genesis:", e);
      const ge = $("#fl-genesis");
      if (ge) ge.innerHTML = `<div class="dim pad">genesis status unavailable: ${esc(e.message)} — <a href="#" onclick="loadFormulas();return false">retry</a></div>`;
    }
    // live series rings: backfill from the GRC history, then every
    // snapshot pushes the current raw — charts read a LONG, live series
    // instead of6 stale points (old-app style jagged tiles)
    S.fvBackfill = fh.core || {};
    if (!S.fvRings || S.fvRingKey !== `${S.symbol}|${S.focusTf}`) {
      S.fvRingKey = `${S.symbol}|${S.focusTf}`;
      S.fvRings = {};
      for (const k of Object.keys(S.fvBackfill)) {
        S.fvRings[k] = (S.fvBackfill[k] || []).slice(-100);
      }
    }
    drawSparks(S.fvBackfill);
  } catch (e) { toast("formulas: " + e.message, "err"); }
}
function drawSparks(core) {
  $$("#fl-cards canvas.spark").forEach(cv => {
    const fid = cv.dataset.f;
    // live ring first (keeps moving between GRCs), history as fallback
    const series = (S.fvRings?.[fid]?.length >= 3 ? S.fvRings[fid] : null)
      || core[fid] || [];
    if (series.length < 3) return;
    const w = cv.clientWidth || 220, h = 46;
    cv.width = w * 2; cv.height = h * 2;         // HiDPI (was: stretched x only)
    const g = cv.getContext("2d");
    g.setTransform(2, 0, 0, 2, 0, 0);
    g.clearRect(0, 0, w, h);
    const win = series.slice(-120);
    // robust range: an outlier must not flatten the trace (user: 'too flat')
    const sorted = [...win].sort((a2, b2) => a2 - b2);
    let mn = sorted[0], mx = sorted[sorted.length - 1];
    if (sorted.length >= 8) {
      const pmn = sorted[Math.floor(sorted.length * 0.05)];
      const pmx = sorted[Math.min(sorted.length - 1,
        Math.ceil(sorted.length * 0.95) - 1)];
      // issue 5: values move only at GRCs, so 90 % of ring samples are
      // identical — the percentile range then collapses to ~0 and the
      // chart drew a dead-flat line while the ticker showed movement.
      // Use the robust range only when it actually contains movement;
      // otherwise keep the full min/max (a couple of jumps must show).
      if (pmx - pmn > 1e-12) { mn = pmn; mx = pmx; }
    }
    if (mx - mn < 1e-12) { mn -= 0.5; mx += 0.5; }   // dead-flat series: centre it
    const rng = (mx - mn) || 1;
    const pad = rng * 0.08;
    mn -= pad; mx += pad;
    const X = i => (i / (win.length - 1)) * (w - 2) + 1;
    const Y = v => h - 3 - ((v - mn) / (mx - mn)) * (h - 6);
    // zero baseline (dotted) — amplitude context
    if (mn < 0 && mx > 0) {
      g.strokeStyle = "rgba(216,216,232,.25)";
      g.setLineDash([3, 3]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(w, Y(0)); g.stroke();
      g.setLineDash([]);
    }
    const rising = win[win.length - 1] >= win[0];
    const col = Math.abs(win[win.length - 1]) < 1e-12 ? "#6e6e8c"
      : rising ? "#22c98a" : "#f0385a";
    // area + line (straight segments — no smoothing)
    g.beginPath();
    win.forEach((v, i) => { i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); });
    g.strokeStyle = col; g.lineWidth = 1.4; g.stroke();
    g.lineTo(X(win.length - 1), h - 1); g.lineTo(X(0), h - 1); g.closePath();
    g.fillStyle = rising ? "rgba(34,201,138,.10)" : "rgba(240,56,90,.10)";
    g.fill();
    // last value dot
    const lv = win[win.length - 1];
    g.fillStyle = col;
    g.beginPath(); g.arc(X(win.length - 1), Y(lv), 2.1, 0, Math.PI * 2); g.fill();
  });
}

/* Side-aware wording for the dominant channel. A state channel reading
   0.04 is 92 % away from its 0.5 neutral on the LOW side — calling that
   "the dominant emotion" without saying which side made the panel announce
   the *absence* of a state (e.g. "Trend exhaustion" when exhaustion was 0). */
function domSideTxt(dom) {
  if (!dom || !dom.side) return "from";
  if (dom.side === "low") return "below";
  if (dom.side === "high") return "above";
  return "at";
}
/* ═══════════ EMOTION ═══════════ */
async function loadEmotion() {
  try {
    const sys = await api(`/api/systems?symbol=${S.symbol}`);
    const em = sys.systems?.[`${S.symbol}|${S.focusTf}`]?.emotion || {};
    // DOMINANT NOW — server-ranked by deviation-from-neutral (the old
    // max(raw value) crowned the first sig()-saturated channel — E2 —
    // on every tie, so "Greed" won forever). Server also tracks the
    // real held-time; the local calc below is only a fallback and now
    // uses the same deviation ranking.
    try {
      const dom = $("#em-dominant");
      const ev = em.emotions || {};
      let bi = 0, best = -1, ri = 0, second = -1, heldS = 0;
      if (em.dominant && em.dominant.id) {
        bi = em.dominant.id;
        best = em.dominant.intensity;
        ri = (em.dominant.runner_up || {}).id || 0;
        second = (em.dominant.runner_up || {}).intensity ?? -1;
        heldS = Math.round(em.dominant.held_s || 0);
      } else {
        for (let i = 1; i <= 21; i++) {
          const dev = Math.abs(Number(ev[i] ?? 0.5) - 0.5) * 2;
          if (dev > best) { second = best; ri = bi; best = dev; bi = i; }
          else if (dev > second) { second = dev; ri = i; }
        }
        if (!S.domEm || S.domEm.i !== bi) S.domEm = { i: bi, since: Date.now() };
        heldS = Math.max(0, Math.round((Date.now() - S.domEm.since) / 1000));
      }
      // Everything below is measured ONCE PER GRC PASS, so it is
      // bit-identical between passes while the panels repaint every 2 s.
      // Print the age next to the headline: a reading that is 3 minutes
      // old is fine, but it must not *look* like it just changed.
      const ageEl = $("#em-age");
      if (ageEl) {
        const at = Number(em.measured_at || 0);
        if (at > 0) {
          const ageS = Math.max(0, Math.round(Date.now() / 1000 - at));
          ageEl.textContent = ageS < 90
            ? `measured ${ageS}s ago · recomputed once per GRC pass`
            : `measured ${Math.round(ageS / 60)}m ago · recomputed once per GRC pass `
              + "(the tape moved since — these channels have not been re-read)";
        } else {
          ageEl.textContent = "";
        }
      }
      if (dom && bi) {
        const pct = Math.round(best * 100);
        const names = em.model?.channels || {};
        const nm = (i) => names[String(i)]?.name || (typeof EMO !== "undefined" && EMO[i - 1]) || ("E" + i);
        const pe = em.primary_emotion || null;
        const fg = em.fear_greed || null;
        // A composite index that only measures part of its own weighting must
        // say so: printing 53 "Neutral" while 50% of the published inputs are
        // missing is the same lie as reporting 0.50 with no input at all.
        const fgCov = Math.round((fg?.coverage ?? fg?.measured_weight ?? 0) * 100);
        // `unmeasured` is a name → reason map, not a list
        const fgMiss = Object.keys(fg?.unmeasured || {}).join(", ");
        const fgHtml = !fg ? "" :
          `<div class="kv"><span>fear &amp; greed</span><b>${fmt(fg.index, 1)} · ${esc(fg.label)} ` +
          (fg.partial
            ? `<span class="acc-warn">(partial — ${fgCov}% of the published weighting has an input here` +
              (fgMiss ? `; missing ${esc(fgMiss)}` : "") + `)</span>`
            : `<span class="dim">(full composite)</span>`) +
          (fg.components
            ? ` <span class="dim">(vol ${fmt(fg.components.volatility, 0)} · momentum ${fmt(fg.components.momentum_volume, 0)})</span>`
            : "") +
          `</b></div>` +
          (fg.published_index != null
            ? `<div class="kv"><span>published F&amp;G</span><b>${fmt(fg.published_index, 1)}` +
              (fg.index != null ? ` <span class="dim">(model is ${fmt(fg.index - fg.published_index, 1)} away)</span>` : "") +
              `</b></div>`
            : "");
        const ca = em.core_affect || null;
        const at = em.action_tendency || null;
        const pad = em.pad || null;
        dom.innerHTML = `
          <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap">
            <div style="font:700 26px/1 var(--mono);color:var(--gold)">${esc((pe?.name || nm(bi)).replace(/^./, c => c.toUpperCase()))}</div>
            <div class="dim">${pe ? `${Math.round((pe.intensity || 0) * 100)}% prototype match (Ekman 1992) · runner-up ${esc((pe.runner_up || {}).name || "—")} ${Math.round((pe.runner_up?.intensity || 0) * 100)}%` : `channel ${esc(nm(bi))} · ${pct}% ${domSideTxt(em.dominant)} neutral`} · held ${heldS}s${em.dominant && em.dominant.directional === false ? ' · <span class="acc-warn">no channel carries a direction this window — this is amplitude, not a state</span>' : ""}${em.dominant?.tied ? ` · <span class="acc-warn">+${em.dominant.tied} channel${em.dominant.tied > 1 ? "s" : ""} tied at this extreme (saturated tape — no single winner)</span>` : ""}</div>
          </div>
          <div class="bar" style="margin-top:8px;max-width:420px"><i class="pos" style="width:${clamp(pe ? (pe.intensity || 0) * 100 : pct, 2, 100)}%"></i></div>
          <div class="kv-row" style="margin-top:8px">
            ${ca ? `<div class="kv"><span>valence (Russell 1980)</span><b>${(ca.valence >= 0 ? "+" : "") + fmt(ca.valence, 2)}</b></div>` : ""}
            ${ca ? `<div class="kv"><span>arousal</span><b>${fmt(ca.arousal, 2)}</b></div>` : ""}
            ${pad ? `<div class="kv"><span>dominance (PAD)</span><b>${fmt(pad.dominance, 2)}</b></div>` : ""}
            ${fgHtml}
            ${at ? `<div class="kv"><span>action tendency (Frijda)</span><b>${esc(at.name)} · urge ${fmt(at.urge, 2)}</b></div>` : ""}
          </div>
          ${fg?.vs_affect ? `<div class="hint acc-warn">${esc(fg.vs_affect)}</div>` : ""}
          <div class="hint">${pe ? "Ekman (1992) basic emotions scored as prototype distance in (valence, arousal, dominance) — a real emotion read, not a fixed id. " : ""}confidence modifier only — never the direction (advice §5.1)</div>`;
      }
    } catch (e) { /* dominant header optional */ }
    $("#em-scope").textContent = `${S.symbol} · ${S.focusTf.toUpperCase()}`;
    $("#em-narrative").innerHTML = `
      <div class="why-reason">${em.narrative || "waiting for the first emotion pass…"}</div>
      <div class="kv-row">
        <div class="kv"><span>net emotion (signed weighted mean, −1…+1)</span><b class="${(em.net_emotion ?? 0) > 0.05 ? "acc-up" : (em.net_emotion ?? 0) < -0.05 ? "acc-dn" : "dim"}">${em.net_emotion == null ? "—" : (em.net_emotion >= 0 ? "+" : "") + fmt(em.net_emotion, 3)}</b></div>
        <div class="kv"><span>CERC agreement sum (spec)</span><b>${fmt(em.cerc, 4)}</b></div>
        <div class="kv"><span>breadth / agreement</span><b>${em.breadth ? `${em.breadth.bull}▲ ${em.breadth.bear}▼ ${em.breadth.flat}·` : "—"} · ${Math.round((em.agreement ?? 0) * 100)}% has a direction</b></div>
        <div class="kv"><span>aggregate intensity</span><b>${fmt(em.aggregate, 4)}</b></div>
        <div class="kv"><span>RTP trauma</span><b>${fmt(em.rtp, 3)}${(em.rtp || 0) > 0.6 ? " ⚠" : ""}</b></div>
        <div class="kv"><span>TEDR → P6</span><b>${fmt(em.tedr, 4)}</b></div>
      </div>
      ${em.features ? `<div class="kv-row"><div class="kv"><span>measured features (tape only — never the signal, G-3)</span>
        <b>samples ${em.features.samples ?? "—"} · vol-z ${fmt(em.features.vol_z, 3)} · climax ×${fmt(em.features.climax_ratio, 2)} · violence gate ${fmt(em.features.violence_gate, 2)} · streak ${em.features.window_streak ?? "—"}</b></div></div>` : ""}
      <div class="hint">21 model channels feed the CERC composite (E22). <b>Net emotion</b> is the sign-weighted mean of the channels that express a direction (Σ wᵢeᵢdᵢ / Σ wᵢeᵢ, ∈ −1…+1): a split crowd reads near zero while the breadth line shows the split. The spec's raw CERC agreement sum is shown beside it unchanged and drives the ±15 % confidence modulator (never the direction, advice §5.1). Circles: conic ring = |value − that channel's own neutral|, border = direction; hover a circle for its citation.</div>
      <div class="kv-grid" style="margin:8px 0">
        <div class="kv"><span>RETAIL emotion (herd/crowd/news channels)</span><b>${em.retail ? `${em.retail.score >= 0 ? "+" : ""}${fmt(em.retail.score, 3)} · ${esc(em.retail.label)}` : "—"}</b></div>
        <div class="kv"><span>HEDGE / smart-money emotion (absorption/divergence)</span><b>${em.hedge ? `${em.hedge.score >= 0 ? "+" : ""}${fmt(em.hedge.score, 3)} · ${esc(em.hedge.label)}` : "—"}</b></div>
        <div class="kv"><span>crowd split (retail − hedge)</span><b>${em.retail && em.hedge ? fmt(em.retail.score - em.hedge.score, 3) + (Math.abs(em.retail.score - em.hedge.score) > 0.3 ? " · crowds DISAGREE" : " · crowds aligned") : "—"}</b></div>
      </div>`;
    const grid = $("#em-grid");
    if (!grid.children.length) {
      for (let i = 1; i <= 22; i++) {
        const c = document.createElement("div");
        c.className = "em-cell"; c.id = "ec-" + i;
        c.innerHTML = `<div class="em-alias">${EMO[i - 1] || ""}</div><div class="em-id">E${i}</div>
          <div class="em-ring"><i>—</i></div><div class="em-tag">—</div>`;
        grid.appendChild(c);
      }
    }
    const channelMeta = em.model?.channels || null;
    for (let i = 1; i <= 22; i++) {
      const v = em.emotions?.[i] ?? 0;
      const dir = em.directions?.[i] ?? 0;
      const cell = $("#ec-" + i);
      const meta = channelMeta?.[String(i)];
      // channels do NOT share a neutral point: the Ekman basics sit at 0.0,
      // the affect/state channels at 0.5. Everything below is measured
      // against the channel's OWN neutral (server-sent).
      const neutral = meta?.neutral ?? (i >= 6 && i <= 11 ? 0.0 : 0.5);
      const dev = Math.abs(v - neutral);
      cell.className = "em-cell " + (dir > 0 ? "bullish" : dir < 0 ? "bearish" : "");
      // an exact 0.00 is an EVENT gate, not a dead channel — say why
      const zr = em.zero_reasons?.[String(i)] || em.zero_reasons?.[i];
      cell.title = (v === 0 && zr) ? zr
        : (v === 0 ? "no event this window — channel abstains" : "");
      const etick = (sys.systems?.[`${S.symbol}|${S.focusTf}`]?.ticks?.emotions || {})[String(i)];
      if (etick) cell.title = ((cell.title ? cell.title + " · " : "") +
        (etick.on ? "✓ contributed (|" + etick.v + "| > 0.01)" : "✗ did not contribute this window"));
      // visible reason + cause for every 0.00 (hover does not exist on mobile)
      let ew = cell.querySelector(".em-why");
      const ecause = (em.zero_causes || {})[String(i)];
      if (v === 0) {
        if (!ew) { ew = document.createElement("div"); ew.className = "em-why"; cell.appendChild(ew); }
        ew.textContent = (ecause ? (ZCAUSE[ecause] || ecause) + " · " : "") +
          (zr || "computed — exactly 0.0 this window");
      } else if (ew) ew.remove();
      const alias = $(".em-alias", cell);
      if (alias) alias.textContent = meta?.name || EMO[i - 1] || "";
      if (meta?.cite) cell.dataset.cite = meta.cite;
      cell.title = (cell.title ? cell.title + " · " : "") +
        (meta ? `${meta.name}${meta.cite ? " (" + meta.cite + ")" : ""} · neutral ${fmt(neutral, 2)}` : "");
      const ring = $(".em-ring", cell);
      ring.style.setProperty("--p", `${clamp(dev * 100, 0, 100)}%`);
      $("i", ring).textContent = (v >= 0 ? "+" : "") +
        (v === 0 ? "0.00" : fx(v, 2));   //0.004 must print0.004, not 0.00
      const tag = $(".em-tag", cell);
      let label = "NEUTRAL";
      if (dev > 0.35) label = "EXTREME"; else if (dev > 0.15) label = "ELEVATED";
      tag.textContent = label; tag.className = "em-tag " + label;
      // live weight + predictive accuracy (accuracy-driven auto-learning)
      const wv = (em.weights || [])[i - 1];
      const av = (em.accuracy || [])[i - 1];
      let wl = $(".em-w", cell);
      if (!wl) { wl = document.createElement("div"); wl.className = "em-w"; cell.appendChild(wl); }
      if (wv != null) wl.innerHTML = `W ${fmt(wv, 3)}` +
        (av != null ? ` · <b class="${av >= 0.55 ? "acc-up" : av <= 0.45 ? "acc-dn" : ""}">${Math.round(av * 100)}%</b>` : "") +
        (em.n_updates != null ? ` <span class="dim">n${em.n_updates}</span>` : "");
    }
  } catch (e) { toast("emotion: " + e.message, "err"); }
}

/* ═══════════ PHYSICS ═══════════ */
async function loadPhysics() {
  try {
    const sys = await api(`/api/systems?symbol=${S.symbol}`);
    const s = sys.systems?.[`${S.symbol}|${S.focusTf}`] || {};
    const ph = s.physics || {};
    $("#ph-scope").textContent = `${S.symbol} · ${S.focusTf.toUpperCase()}`;
    $("#ph-mult").innerHTML = `
      <div class="kv"><span>UFP (P22 aggregate)</span><b>${fmt(ph.ufp, 4)}</b></div>
      <div class="kv"><span>WFCP ×(0.5+0.5·x)</span><b>${fmt(ph.wfcp, 4)}</b></div>
      <div class="kv"><span>DRPT ×(1−x)</span><b>${fmt(ph.drpt, 4)}</b></div>
      <div class="kv"><span>GSPI ×(1−x)</span><b>${fmt(ph.gspi, 4)}</b></div>
      <div class="kv"><span>TME → CCI id 99</span><b>${fmt(ph.tme, 4)}</b></div>
      <div class="kv"><span>alpha mixing</span><b>${fmt(Object.values(ph.alpha || {})[0], 4)}</b></div>`;
    const bgn = (S.snap?.bgn || {})[S.symbol] || {};   // selected asset — no asset is hard-coded
    $("#ph-bgn").innerHTML = `<div class="cell h">β</div>` +
      TF.map(t => `<div class="cell h">${t.toUpperCase()}</div>`).join("") +
      TF.map(t => `<div class="cell h">${t.toUpperCase()}</div>` + TF.map(t2 =>
        `<div class="cell" style="color:${(bgn[t]?.[t2] ?? 0) >= 0 ? "var(--cyan)" : "var(--red)"}">${fmt(bgn[t]?.[t2] ?? 0, 3)}</div>`).join("")).join("");
    const grid = $("#ph-grid"); grid.innerHTML = "";
    const vals = ph.values || {};
    const det = ph.details || {};
    // §6.2: active/inactive table WITH reasons (F-7: never a silent 0.00)
    const mech = ph.mechanisms || [];
    const mechEl = $("#ph-mech");
    if (mechEl && mech.length) {
      mechEl.innerHTML = `<div class="kv-row">` + mech.map(m =>
        `<span class="chip ${m.active ? "chip-ok" : "chip-warn"}"
          title="${m.active ? esc((m.logic || []).join(" · ")) : esc((m.cause ? "[" + m.cause + "] " : "") + (m.why || ""))}"
          >P${m.id} ${m.active ? "●" : "○"} ${fx(m.score, 3)}</span>`).join("") +
        `</div><div class="hint">${mech.filter(m => !m.active).slice(0, 4).map(m =>
          `P${m.id} ${m.cause ? `<b class="acc-warn">[${esc(m.cause)}]</b>` : ""}: ${esc(m.why)}`).join(" · ") || "all mechanisms active"}</div>`;
    }
    // §6.1 estimators print their derivation
    const est = ph.estimators || {};
    const estEl = $("#ph-est");
    if (estEl && est.ready) {
      const blocks = ["kramers_moyal", "ising", "fdt", "maxent"].filter(k => est[k]).map(k => {
        const e = est[k];
        const nums = Object.entries(e).filter(([kk, vv]) => kk !== "derivation")
          .map(([kk, vv]) => `${kk} ${Array.isArray(vv) ? vv.map(x => x == null ? "—" : fmt(x, 5)).join(",") : (vv == null ? "—" : fmt(vv, 5))}`)
          .join(" · ");
        return `<div class="logic"><b>${k.replace("_", " ").toUpperCase()}</b> — ${nums}
          <div class="dim">${esc(e.derivation || "")}</div></div>`;
      }).join("");
      estEl.innerHTML = blocks || `<div class="dim">${esc(est.note || "estimators not ready")}</div>`;
    }
    for (let p = 1; p <= 22; p++) {
      const v = p === 22 ? ph.ufp : vals[String(p)];
      if (v == null) continue;
      const card = document.createElement("div");
      card.className = "card";
      let extra = "";
      if (p === 1 && det.p1) extra = `<div class="logic">predicted ${px(det.p1.p_predicted)} · σ ${fmt(det.p1.sigma_q, 1)} · Q ${fmt(det.p1.confidence_q, 2)}</div>`;
      if (p === 15 && det.p15) {
        const d15 = det.p15;
        extra = `<div class="logic">Doppler ${fmt(d15.doppler, 3)} · β ${fmt(d15.beta, 4)} · leader ${esc(String(d15.leader || "—"))} · momentum ${fmt(d15.momentum_raw, 3)} · ${d15.blueshift ? "blueshift (approaching)" : "redshift (receding)"}</div>`;
      }
      const aW = (ph.alpha || {})[String(p)];
      const aA = (ph.accuracy || ph.alpha_accuracy || {})[String(p)];
      const accLine = aW != null
        ? `<div class="kv-row"><div class="kv"><span>α weight</span><b>${fmt(aW, 4)}</b></div>` +
          (aA != null ? `<div class="kv"><span>accuracy</span><b class="${aA >= 0.55 ? "acc-up" : aA <= 0.45 ? "acc-dn" : ""}">${Math.round(aA * 100)}%</b></div>` : "") +
          `</div>`
        : "";
      const why = (ph.mechanisms || []).find(mm => mm.id === p);
      const whyLine = why
        ? `<div class="hint" title="${esc(why.logic?.join(" · ") || "")}">${why.active ? "✓" : "○ why: "}${why.cause ? `<b class="acc-warn">[${ZCAUSE[why.cause] || why.cause}]</b> ` : ""}${esc(why.why || "")}</div>`
        : "";
      card.innerHTML = `<h4><span>P${p} · ${P_NAMES[p] || ""}</span><span>${v >= 0 ? "+" : ""}${fx(v, 4)}</span></h4>
        <div class="kv-row"><div class="kv"><span>value</span><b style="color:${v >= 0 ? "var(--green)" : "var(--red)"}">${v >= 0 ? "▲" : "▼"} ${fx(v, 4)}</b></div></div>
        ${accLine}
        ${whyLine}${extra}<div class="logic">${P_LOGIC[p] || ""}</div>`;
      grid.appendChild(card);
    }
  } catch (e) { toast("physics: " + e.message, "err"); }
}

/* ═══════════ NEWS ═══════════ */
const NIV_HIST = [];          // tf1 NIV ring for the 90 s trace (docs11 §6)
let NIV_LAST_TS = 0;

async function loadNews() {
  try {
    const d = await api(`/api/news/${S.symbol}`);
    // docs11: coverage header, source lines, dropped examples, emergencies
    try {
      const st = await api("/api/news/status");
      $("#nw-coverage").textContent = st.coverage || "—";
      $("#nw-plan").textContent =
        `plan ${st.plan || "?"} · ${st.keys || 0} key(s) · poll ${st.poll_interval_s || "?"}s`;
      $("#nw-sources").innerHTML = (st.sources || []).map(x =>
        `<div class="kv"><span>${esc(x.source)} t${x.tier}</span><b>${x.state}` +
        `${x.backoff_level ? " L" + x.backoff_level : ""}` +
        `${x.dropped_429 ? " · 429×" + x.dropped_429 : ""}</b></div>`).join("");
      const ex = st.filtered_examples || [];
      // §5.1 drop accounting: last window / cumulative / the reasons behind
      // them. "0 dropped" while the ingest gate binned half the wire was
      // the deceptive version of this line.
      const reasons = Object.entries(st.dropped_reasons || {})
        .map(([k, v]) => `${v}× ${k}`).slice(0, 3).join(" · ");
      $("#nw-dropped").textContent =
        `dropped ${st.last_dropped_count || 0} since last poll · ` +
        `${st.total_dropped_count || 0} total` +
        (reasons ? " — " + reasons : "") +
        (ex.length ? " — e.g. " + ex.map(h => `"${h}"`).join("; ") : "");
      const em = st.emergency_history || [];
      $("#nw-emerg").innerHTML = em.length
        ? em.slice().reverse().map(r =>
            `<div class="kv"><span>${new Date(r.ts * 1000).toLocaleTimeString()} ` +
            `${esc(r.asset)} ${esc(String(r.lock_id || ""))}</span>` +
            `<b title="${esc(r.reason || "")}">${fmt(r.old_entry, 2)} → ${fmt(r.new_entry, 2)}</b></div>`).join("")
        : `<span class="dim">none</span>`;
      // per-horizon chips from the same status payload
      const hz = (st.horizons || {})[S.symbol] || {};
      $("#nw-horizons").innerHTML = TF.map(tf => {
        const h = hz[tf] || {};
        // honest empty state: 0.00 only prints when items actually scored.
        // In simulator mode the panel still shows what the sim is saying,
        // explicitly marked as not scored.
        if ((h.n_items || 0) === 0) {
          if (h.simulated_n)
            return `<div class="kv"><span>${tf.toUpperCase()} <span class="dim">sim</span></span>` +
              `<b class="dim">SIM ${fmt(h.simulated_impact, 3)} · ${h.simulated_n} simulated headline(s)` +
              ` — not scored, cannot move a trade</b></div>`;
          return `<div class="kv"><span>${tf.toUpperCase()}</span>` +
            `<b class="dim">— <span class="dim">no live items (synthetic never scores)</span></b></div>`;
        }
        return `<div class="kv"><span>${tf.toUpperCase()} ` +
          `<span class="dim">½${Math.round(h.half_life_min || 0)}m</span></span>` +
          `<b style="color:${(h.impact || 0) >= 0 ? "var(--green)" : "var(--red)"}">` +
          `${fmt(h.impact, 3)} <span class="dim">niv ${fmt(h.niv, 3)} smd ${fmt(h.smd, 3)}</span></b></div>`;
      }).join("");
      const h1 = hz.tf1;
      const liveNiv = (d.live || {}).niv_live;
      // SEED the 90 s trace from the server's ring: the chart used to start
      // from a single sample, which renders as a dead flat line.
      const ring = (d.live || {}).ring;
      if (Array.isArray(ring) && ring.length > 1) {
        NIV_HIST.length = 0;
        ring.slice(-60).forEach(([t, v]) => {
          const ts = Number(t) * (Number(t) < 1e12 ? 1000 : 1);
          if (Number.isFinite(ts) && Number.isFinite(Number(v)))
            NIV_HIST.push({ t: ts, v: Number(v) });
        });
        drawNiv();
      }
      if (liveNiv !== undefined) pushNiv(liveNiv);
      else if (h1 && h1.niv !== undefined) pushNiv(h1.niv);
    } catch (e) { /* status optional */ }
    $("#nw-scope").textContent = `${S.symbol} · ${d.source || ""} · ${d.count ?? (d.items || []).length} items`;
    const cats = d.categories || {};
    // §5.1 the AGGREGATE row is the horizon-weighted mean of the four TF
    // impacts, so it can never contradict the rows directly above it.
    const aggImp = d.aggregate_impact ?? d.aggregate ?? 0;
    $("#nw-agg").innerHTML = TF.map(tf => {
      const imp = (d.impacts || {})[tf];
      return `<div class="kv"><span>${tf.toUpperCase()} impact</span><b>${fmt(imp, 4)}</b></div>`;
    }).join("") +
      `<div class="kv"><span>aggregate <span class="dim">h-weighted</span></span>` +
      `<b style="color:${aggImp >= 0 ? "var(--green)" : "var(--red)"}">${fmt(aggImp, 4)}</b></div>
      <div class="kv"><span>polarity</span><b>${fmt(d.polarity, 3)}</b></div>
      <div class="kv"><span>NIV live <span class="dim">decayed</span></span>` +
      `<b>${fmt((d.live || {}).niv_live ?? d.niv, 4)}` +
      ` <span class="dim">base ${fmt((d.live || {}).niv_base ?? d.niv, 4)}</span></b></div>`;
    const tb = $("#nw-table tbody"); tb.innerHTML = "";
    (d.items || []).forEach(it => {
      const cat = cats[it.category] || {};
      const age = Math.max(0, (Date.now() / 1000 - (it.ts || 0)) / 60);
      const decay = Math.max(0, (cat.duration || 60) - age);
      const tier = it.reliability >= 0.8 ? "GOLD" : it.reliability >= 0.5 ? "SILVER" : "BRONZE";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${new Date((it.ts || 0) * 1000).toLocaleTimeString()}</td>
        <td style="font-family:inherit">${esc(it.headline)}</td>
        <td class="tier-${tier}">${tier} ${fmt(it.reliability, 2)}</td>
        <td>${it.category} <span class="dim">×${fmt(cat.weight, 2)}</span></td>
        <td style="color:${it.sentiment >= 0 ? "var(--green)" : "var(--red)"}">${it.sentiment >= 0 ? "+" : ""}${fmt(it.sentiment, 2)}</td>
        <td>${fmt(it.impact, 3)}</td>
        <td class="dim">${decay > 0 ? Math.floor(decay) + "m left" : "expired"}</td>`;
      tb.appendChild(tr);
    });
    if (!(d.items || []).length) tb.innerHTML = `<tr><td colspan="7" class="dim">no items</td></tr>`;
  } catch (e) { toast("news: " + e.message, "err"); }
}

let NEWS_LAST_COUNT = 0;
let NEWS_LAST_TS = 0;
function onNewsTicker(msg) {
  // docs11 §6: 4-second publisher — update chips live, feed the NIV trace,
  // and STREAM new headlines into an open News tab without a manual refresh
  const newest = (msg.latest && msg.latest.length)
    ? Number(msg.latest[0].ts || 0) : 0;
  if (S.view === "news" &&
      (msg.count !== NEWS_LAST_COUNT || newest > NEWS_LAST_TS)) {
    NEWS_LAST_COUNT = msg.count ?? NEWS_LAST_COUNT;
    NEWS_LAST_TS = Math.max(NEWS_LAST_TS, newest);
    loadNews();
  }
  const a = (msg.assets || {})[S.symbol];
  if (a && S.view === "news") {
    const hz = a.horizons || {};
    $("#nw-horizons").innerHTML = TF.map(tf => {
      const h = hz[tf] || {};
      if ((h.n_items || 0) === 0) {
        if (h.simulated_n)
          return `<div class="kv"><span>${tf.toUpperCase()} <span class="dim">sim</span></span>` +
            `<b class="dim">SIM ${fmt(h.simulated_impact, 3)} · ${h.simulated_n} simulated — not scored</b></div>`;
        return `<div class="kv"><span>${tf.toUpperCase()}</span>` +
          `<b class="dim">— <span class="dim">no live items</span></b></div>`;
      }
      return `<div class="kv"><span>${tf.toUpperCase()}</span>` +
        `<b style="color:${(h.impact || 0) >= 0 ? "var(--green)" : "var(--red)"}">` +
        `${fmt(h.impact, 3)} <span class="dim">niv ${fmt(h.niv, 3)} smd ${fmt(h.smd, 3)}</span></b></div>`;
    }).join("");
    if (a.niv_live !== undefined) pushNiv(a.niv_live);
    else if (hz.tf1 && hz.tf1.niv !== undefined) pushNiv(hz.tf1.niv);
    if (a.emergency) toast("⚡ auto emergency re-lock fired", "err");
    $("#nw-coverage").textContent = a.coverage || $("#nw-coverage").textContent;
  }
}

function pushNiv(v) {
  const now = Date.now();
  if (now - NIV_LAST_TS < 500) return;      // coalesce (chart-lag lesson)
  NIV_LAST_TS = now;
  NIV_HIST.push({ t: now, v });
  while (NIV_HIST.length > 60) NIV_HIST.shift();
  drawNiv();
}

function drawNiv() {
  const c = $("#nw-niv"); if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 300, h = 54;
  if (c.width !== w * dpr || c.height !== h * dpr) {
    c.width = w * dpr; c.height = h * dpr;
  }
  const x = c.getContext("2d");
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, w, h);
  x.strokeStyle = "rgba(120,140,170,.45)";
  x.beginPath(); x.moveTo(0, h / 2); x.lineTo(w, h / 2); x.stroke();
  if (NIV_HIST.length < 2) return;
  const vs = NIV_HIST.map(p => p.v);
  // TRUE range: a fixed 0.05 ceiling flattened a ±0.003 NIV into a
  // straight line. Scale to the data, with a 2 bp floor against noise.
  const lo = Math.min(...vs), hi = Math.max(...vs);
  const mid = (hi + lo) / 2;
  const span = Math.max(0.002, hi - lo);
  const yOf = v => h / 2 - ((v - mid) / span) * (h * 0.42);
  x.strokeStyle = "#4fd1c5"; x.lineWidth = 1.5; x.beginPath();
  NIV_HIST.forEach((p, i) => {
    const px = (i / (NIV_HIST.length - 1)) * w;
    const py = yOf(p.v);
    i ? x.lineTo(px, py) : x.moveTo(px, py);
  });
  // label the scale so a flat-looking trace is never ambiguous
  x.fillStyle = "rgba(147,160,184,.85)";
  x.font = "9px monospace";
  x.fillText(`+${hi.toFixed(4)}`, 4, 9);
  x.fillText(`${lo.toFixed(4)}`, 4, h - 3);
  x.stroke();
}

async function drillEmergency() {
  if (!confirm(`Emergency drill for ${S.symbol}? Re-locks at today's price, ` +
               `halts new locks 30 s (original entry kept in the reason).`)) return;
  try {
    const out = await api("/api/news/emergency", {
      method: "POST", body: JSON.stringify({
        asset: S.symbol, tf: S.tf || "tf1",
        reason: "manual drill from news card" }) });
    toast(out.ok ? `emergency ok — halt ${out.halt_s}s` :
          (out.error || "emergency failed"), out.ok ? "ok" : "err");
    loadNews();
  } catch (e) { toast("drill: " + e.message, "err"); }
}

/* ═══════════ BRAINS ═══════════ */
async function loadBrains() {
  try {
    const [geo, sys, bst] = await Promise.all([
      api("/api/brain-geometry"),
      api(`/api/systems?symbol=${S.symbol}`),
      api("/api/brain/status").catch(() => null),
    ]);
    const s = sys.systems?.[`${S.symbol}|${S.focusTf}`] || {};
    const br = s.brain || {};
    const isReal = String(geo.connectome_source || "").includes("real");
    $("#br-source").textContent = isReal
      ? "REAL · Janelia hemibrain v1.2" : "structural fallback";
    $("#br-source").className = "br-source " + (isReal ? "real" : "sim");
    const cs = geo.connectome_stats || {};
    $("#br-real").innerHTML = `<div class="kv-row">
      <div class="kv"><span>dataset</span><b>${geo.connectome_dataset || "synthetic"}</b></div>
      <div class="kv"><span>neurons (traced)</span><b>${cs.neurons != null ? Number(cs.neurons).toLocaleString() : "—"}</b></div>
      <div class="kv"><span>connections</span><b>${cs.edges != null ? Number(cs.edges).toLocaleString() : "—"}</b></div>
      <div class="kv"><span>synapses</span><b>${cs.synapses != null ? Number(cs.synapses).toLocaleString() : "—"}</b></div>
      <div class="kv"><span>reciprocity</span><b>${cs.reciprocity ?? "—"}</b></div>
      <div class="kv"><span>graph units (this app)</span><b>${geo.units || "—"} · ${geo.edges || "—"} edges</b></div>
    </div>
    <div class="hub-list">${(geo.connectome_hubs || []).slice(0, 8).map(h =>
      `<span class="hub" title="type ${h.type} · degree ${h.total_degree}">${h.type} <i>${h.total_degree}</i></span>`).join("")}</div>
    <div class="hint">Real hemibrain v1.2 (Janelia FlyEM, CC BY 4.0) — hub ordering (APL, DPM, MBON…) and per-unit edge weights follow the measured degree distribution; the four market-brain instances run the spec's activation math on this structure.</div>`;
    $("#br-consensus").innerHTML = `
      <div class="kv"><span>polarity</span><b>${br.polarity > 0 ? "▲ BULL" : br.polarity < 0 ? "▼ BEAR" : "● FLAT"}</b></div>
      <div class="kv"><span>consensus</span><b>${fmt(br.consensus, 4)}</b></div>
      <div class="kv"><span>agreement (${esc(br.agreement_count || "—")} voters)</span><b>${fmt(br.agreement, 3)}</b></div>
      <div class="kv"><span>spread (max − min vote)</span><b>${fmt(br.spread, 3)}</b></div>
      <div class="kv"><span>abstained</span><b>${(br.abstained || []).length ? esc((br.abstained || []).join(", ")) : "none"}</b></div>
      <div class="kv"><span>experience EQ</span><b>${fmt(br.experience, 3)}</b></div>
      <div class="kv"><span>dominant state</span><b>${br.dominant || "—"} (${fmt(br.p_dominant, 2)})${br.state_confident === false ? ' <span class="dim">· posterior flat — no state called</span>' : ""}</b></div>
      <div class="kv"><span>hormones D/O/S/T</span><b>${fmt(br.hormones_avg?.dopamine, 2)} / ${fmt(br.hormones_avg?.octopamine, 2)} / ${fmt(br.hormones_avg?.serotonin, 2)} / ${fmt(br.hormones_avg?.tyramine, 2)}</b></div>`;
    const wrap = $("#br-instances"); wrap.innerHTML = "";
    (br.instances || []).forEach(inst => {
      const card = document.createElement("div");
      card.className = "card";
      const h = inst.hormones || {};
      card.innerHTML = `<h4><span>${inst.id} · ${esc(inst.role || "")}</span><span>${(inst.signal ?? 0) >= 0 ? "+" : ""}${fmt(inst.signal ?? 0, 4)}</span></h4>
        <div class="logic">specialty vote ${fmt(inst.signal ?? 0, 3)} · hormone ${fmt(inst.h_signal, 3)} · input-driven response ${inst.response == null ? "—" : fmt(inst.response, 4)}${inst.input_energy != null ? ` (input energy ${fmt(inst.input_energy, 3)})` : ""}${inst.abstain ? ' · <span class="acc-warn">ABSTAIN — this window did not drive it</span>' : ""}</div>
        <div class="logic">${inst.backend === "flywire-v783-real" ? `🧠 ${Number(inst.neurons).toLocaleString()} real neurons · ${Number(inst.synapses).toLocaleString()} real synapses (FlyWire v783)` : "synthetic graph — real circuit not ready"}</div>
        <div class="kv-row">
          <div class="kv"><span>${inst.id === "DB-3" ? "state" : inst.id === "DB-1" ? "pattern" : inst.id === "DB-2" ? "experience EQ (MB · LH balance)" : "learn thrust"}</span><b>${inst.dominant ?? (inst.id === "DB-1" ? fmt(inst.pattern, 3) : inst.id === "DB-2" ? fmt(inst.experience, 3) : fmt(inst.signal, 3))}</b></div>
          ${inst.id === "DB-2" ? "" : `<div class="kv"><span>experience</span><b>${fmt(inst.experience, 3)}</b></div>`}
          ${inst.id === "DB-1" ? `<div class="kv"><span>approach/avoid selectivity</span><b>${fmt(inst.selectivity, 4)} <span class="dim">(MBONs move as one block on this connectome)</span></b></div>` : ""}
          ${inst.id === "DB-4" ? `<div class="kv"><span>per-formula Δ</span><b>${inst.informative ? esc(Object.entries(inst.weight_adj || {}).slice(0, 4).map(([k, v]) => `F${k} ${v >= 0 ? "+" : ""}${fmt(v, 3)}`).join(" · ")) : `<span class="acc-warn">${esc(inst.informative === false ? "all 35 deltas identical — controller has nothing to differentiate" : "—")}</span>`}</b></div>` : ""}
        </div>
        <div class="kv-row" style="margin-top:8px">
          ${["dopamine", "serotonin", "octopamine", "tyramine"].map(k =>
            `<div class="kv"><span>${k.slice(0, 4)} ${fmt(h[k], 2)}</span><div class="bar"><i class="${(h[k] ?? 0) > 0.5 ? "pos" : "neg"}" style="width:${clamp((h[k] ?? 0) * 100, 0, 100)}%"></i></div></div>`).join("")}
        </div>
        <div class="logic">${inst.id === "DB-1"
            ? `pattern ${fmt(inst.pattern, 3)}${inst.selectivity != null ? ` · selectivity ${fmt(inst.selectivity, 4)}` : ""}`
            : inst.id === "DB-2"
              ? `experience ${fmt(inst.experience, 3)}${inst.response != null ? ` · drive ${fmt(inst.response, 4)}` : ""}`
              : inst.id === "DB-3"
                ? `state ${esc(inst.dominant ?? "—")}${inst.p_dominant != null ? ` · P ${fmt(inst.p_dominant, 2)}${inst.confident === false ? " (shrug — flat posterior)" : ""}` : ""}`
                : `learn thrust ${fmt(inst.signal, 3)} · informed ${inst.informative == null ? "—" : inst.informative === false ? "no (all deltas identical)" : "yes"}`}</div>`;
      wrap.appendChild(card);
    });
    // ── FlyWire v783: chip, five-step verification, polarity, RPE ──
    if (bst && bst.flywire) {
      const fw = bst.flywire;
      const chip = bst.chip || fw.chip || { kind: "absent", text: "brain absent" };
      const cc = chip.kind === "ok" ? "#22c55e"
        : chip.kind === "build" ? "#3b82f6" : "#ef4444";
      const cel = $("#br-chip");
      if (cel) {
        cel.textContent = chip.text;
        cel.style.color = cc;
        cel.style.border = `1px solid ${cc}`;
      }
      $("#br-verify").innerHTML = (fw.verify || []).map(v => {
        // optional sources with no key are neutral, never an error
        // (screenshot: red "no key" beside a healthy connectome)
        const d = String(v.detail || "");
        const neutral = v.ok && (d.includes("optional") || d.includes("no key"));
        return `<div class="kv"><span>${esc(v.step)}${v.ms != null ? ` <span class="dim">${v.ms} ms</span>` : ""}</span>` +
        `<b class="${neutral ? "dim" : v.ok ? "acc-up" : "acc-dn"}" title="${esc(d)}">${esc(d.slice(0, 90))}</b></div>`;
      }).join("") || `<div class="dim">no verification yet</div>`;
      const pol = fw.polarity || {};
      const flips = pol.flips || [];
      const flipped = flips.map((f, i) => f < 0 ? i : -1).filter(i => i >= 0);
      $("#br-polarity").innerHTML = `
        <div class="kv"><span>channels</span><b>${pol.n_channels || 0}</b></div>
        <div class="kv"><span>flipped (innate aversion)</span><b>${flipped.length ? flipped.join(", ") : "none"}</b></div>
        <div class="kv"><span>dead (no circuit response)</span><b>${(pol.dead || []).length ? pol.dead.join(", ") : "none"}</b></div>
        <div class="kv"><span>pass</span><b>${fw.pass?.ms ?? "—"} ms · ${fw.pass?.age_s ?? "—"} s ago · conf ${fw.pass?.conf ?? "—"}</b></div>`;
      const dp = fw.dopamine || {};
      $("#br-dopamine").innerHTML = `
        <div class="kv"><span>E (expectation)</span><b>${fmt(dp.E, 3)}</b></div>
        <div class="kv"><span>δ (last RPE)</span><b>${fmt(dp.delta, 3)}</b></div>
        <div class="kv"><span>drg (tanh 2.25·δ)</span><b>${fmt(dp.drg, 3)}</b></div>
        <div class="kv"><span>α (learn rate)</span><b>${fmt(dp.alpha, 3)}</b></div>
        <div class="kv"><span>tonic / phasic</span><b>${fmt(dp.tonic, 2)} / ${fmt(dp.phasic, 3)}</b></div>`;
      const fm = fw.formulas || {};
      const cell = (name, f) => {
        const v = f || {};
        return `<div class="kv"><span>${name}${v.kind === "gauge" ? " (gauge)" : ""}</span>` +
          `<b class="${v.zero_reasons ? "acc-warn" : ""}" title="${esc(v.zero_reasons || "")}">` +
          `${v.zero_reasons ? "why: " + esc(v.zero_reasons) : fmt(v.value, 4)}</b></div>`;
      };
      $("#br-fw-formulas").innerHTML =
        cell("CCSv2 circuit consensus", fm.ccsv2) + cell("KCAE stability", fm.kcae);
      // ── full connectome anatomy: neurons · cells · synapses · matrix
      const an = fw.anatomy || {};
      const anEl = $("#br-anatomy");
      if (anEl) {
        if (an.state === "ready") {
          const cls = Object.entries(an.classes || {})
            .sort((a, b2) => b2[1] - a[1]);
          const top = cls.slice(0, 12);
          const mx = (an.block?.matrix || []).flat();
          const mmax = Math.max(1e-9, ...mx);
          const bclasses = (an.block?.classes || []);
          const keep = cls.slice(0, 10).map(c => c[0]);
          const ki = keep.map(c => bclasses.indexOf(c)).filter(x => x >= 0);
          const matrix = (an.block?.matrix || []).filter((_, r) => ki.includes(r))
            .map(row => ki.map(ci => row[ci]));
          // NUMERIC, NEVER A PLACEHOLDER (report: "cell class still ---"):
          // a real measured 0 prints 0.00, a missing cell says n/a — the
          // table can no longer look half-empty when it is fully computed.
          const cellv = v2 => {
            const num = Number(v2);
            const ok = Number.isFinite(num);
            const a2 = ok && num > 0 ? (Math.log1p(num) / Math.log1p(mmax)) * 0.7 : 0;
            const txt = !ok ? "n/a"
              : num === 0 ? "0.00"
              : Math.abs(num) < 0.01 ? num.toExponential(1) : num.toFixed(2);
            return `<td style="background:rgba(0,212,240,${a2.toFixed(3)});text-align:right;font-family:var(--mono);font-size:11.5px" title="${ok ? num : "n/a"}">${txt}</td>`;
          };
          const ntc = Object.entries(an.nt_census || an.mbon_nt || {})
            .filter(([k]) => k && k !== "?").slice(0, 6);
          anEl.innerHTML = `
            <div class="kv-row">
              <div class="kv"><span>neurons (cells)</span><b>${(an.neurons || 0).toLocaleString()}</b></div>
              <div class="kv"><span>connections (≥5 syn)</span><b>${(an.connections || 0).toLocaleString()}</b></div>
              <div class="kv"><span>synapses (Σ)</span><b title="${esc(an.synapses_basis || "measured census")}">${an.synapses != null ? Number(an.synapses).toLocaleString() : "unknown — connectome cache has no census"}</b></div>
              <div class="kv"><span>mean degree</span><b>${an.mean_degree ?? "—"}</b></div>
              <div class="kv"><span>density</span><b>${an.density ?? "—"}</b></div>
            </div>
            <div class="hub-list" style="margin-top:8px">${top.map(([c, n]) =>
              `<span class="hub">${esc(c)} <i>${n.toLocaleString()}</i></span>`).join("")}</div>
            <div class="kv-row" style="margin-top:8px">
              ${ntc.map(([k, n]) => `<div class="kv"><span>nt ${esc(k)}</span><b>${n.toLocaleString()}</b></div>`).join("")}
              ${Object.entries(an.sides || {}).filter(([k]) => k && k !== "?").slice(0, 3)
                .map(([k, n]) => `<div class="kv"><span>side ${esc(k)}</span><b>${n.toLocaleString()}</b></div>`).join("")}
            </div>
            <div class="hint">cell_class × cell_class summed |W| (log colour) — top10 classes</div>
            <div style="overflow-x:auto"><table class="blk-matrix"><tr><th></th>${keep.map(c => `<th title="${esc(c)}">${esc(c).slice(0, 8)}</th>`).join("")}</tr>
              ${matrix.map((row, r) => `<tr><th title="${esc(keep[r])}">${esc(keep[r]).slice(0, 8)}</th>${row.map(cellv).join("")}</tr>`).join("")}
            </table></div>
            ${an.top_cell_types && Object.keys(an.top_cell_types).length ? `<div class="hint">top cell types: ${Object.entries(an.top_cell_types).slice(0, 8).map(([t, n]) => `${esc(t)} (${n})`).join(" · ")}</div>` : ""}`;
        } else if (an.state === "computing") {
          anEl.innerHTML = `<div class="kv-row">
              <div class="kv"><span>neurons (cells)</span><b>${Number(an.neurons || 0).toLocaleString()}</b></div>
              <div class="kv"><span>connections (≥5 syn)</span><b>${Number(an.connections || 0).toLocaleString()}</b></div>
              <div class="kv"><span>synapses (Σ)</span><b>${an.synapses != null ? Number(an.synapses).toLocaleString() : "unknown"}</b></div>
              <div class="kv"><span>mean degree</span><b>${an.mean_degree ?? "—"}</b></div>
              <div class="kv"><span>density</span><b>${an.density ?? "—"}</b></div>
            </div>
            <div class="hint">census from the loaded graph — the class × class heatmap finishes in a few seconds…</div>`;
          if (!anEl.dataset.retry) {
            anEl.dataset.retry = "1";
            setTimeout(() => { delete anEl.dataset.retry; loadBrains(); }, 4000);
          }
        } else {
          anEl.innerHTML = `<div class="dim pad">${an.state === "absent" ? "connectome absent — anatomy withheld" : esc(an.error || "unavailable")}</div>`;
        }
      }
      // ── LATEST READ-OUT (the pass the engine used)
      const ro = fw.readout || {};
      const wo = ro.whole, mo = ro.mb;
      const roEl = $("#br-readout");
      if (roEl) {
        if (ro.whole || ro.mb) {
          roEl.innerHTML = `
            <div class="kv-row">
              <div class="kv"><span>whole-brain balance</span><b>${wo ? fx(wo.balance, 4) : "—"}</b></div>
              <div class="kv"><span>approach / avoid (MBON)</span><b>${wo ? fx(wo.approach, 4) : "—"} / ${wo ? fx(wo.avoid, 4) : "—"}</b></div>
              <div class="kv"><span>active Kenyon cells</span><b>${wo ? `${wo.kc_active} / ${fw.anatomy?.classes?.Kenyon_Cell ?? "?"}` : "—"}</b></div>
              <div class="kv"><span>active per hop</span><b>${wo?.hop_counts ? wo.hop_counts.join(" → ") : "—"}</b></div>
              <div class="kv"><span>pass</span><b>${wo?.ms ?? mo?.ms ?? "—"} ms</b></div>
            </div>
            <div class="kv-row">
              <div class="kv"><span>MB balance</span><b>${mo ? fx(mo.balance, 4) : "—"}</b></div>
              <div class="kv"><span>LH (lateral horn)</span><b>${mo ? fx(mo.lh, 4) : "—"}</b></div>
              <div class="kv"><span>MB active KC</span><b>${mo ? `${mo.kc_active} / ${mo.kc_total}` : "—"}</b></div>
              <div class="kv"><span>top MBONs</span><b>${(wo?.top_mbons || []).map(t => `${esc(t.mbon)} ${t.h >= 0 ? "+" : ""}${t.h} (${t.nt.slice(0, 3)})`).join(" · ") || "—"}</b></div>
            </div>`;
        } else {
          roEl.innerHTML = `<div class="dim pad">first whole-brain pass running — read-out in seconds…</div>`;
          if (!roEl.dataset.retry) {
            roEl.dataset.retry = "1";
            setTimeout(() => { delete roEl.dataset.retry; loadBrains(); }, 5000);
          }
        }
      }
      // ── PATHWAYS + CHANNEL MATRIX (from the anatomy pass)
      if (an.state === "ready") {
        const pw = (an.pathways || []).slice(0, 10);
        const pwEl = $("#br-pathways");
        if (pwEl) {
          pwEl.innerHTML = pw.length ? `<div class="logic" style="display:grid;grid-template-columns:1fr auto auto;gap:4px 12px">` +
            pw.map(x => `<span>${esc(x.frm)} → ${esc(x.to)}</span>` +
              `<span style="color:${x.drive >= 0 ? "var(--green)" : "var(--red)"};text-align:right">${x.drive >= 0 ? "+" : ""}${fx(x.drive, 3)}</span>` +
              `<span class="dim">${x.sign}</span>`).join("") + `</div>`
            : `<div class="dim pad">no signed pathways yet</div>`;
        }
        const cm = an.channel_matrix;
        const cmEl = $("#br-chan-matrix");
        if (cm && cmEl) {
          const cmap = bst.channel_map || [];
          // columns: top classes by total |drive|
          const kx = cm.classes.map((c2, ci) => ci)
            .sort((a2, b2) =>
              Math.abs(cm.matrix.reduce((s2, r) => s2 + Math.abs(r[b2] || 0), 0)) -
              Math.abs(cm.matrix.reduce((s2, r) => s2 + Math.abs(r[a2] || 0), 0)))
            .slice(0, 9);
          const mmax = Math.max(1e-9, ...cm.matrix.flat().map(Math.abs));
          const cellv = v2 => {
            const al2 = v2 !== 0 ? Math.min(0.85, (Math.log1p(Math.abs(v2)) / Math.log1p(mmax)) * 0.85) : 0;
            const bg = v2 >= 0 ? `rgba(34,201,138,${al2.toFixed(3)})` : `rgba(240,56,90,${al2.toFixed(3)})`;
            const num = Number(v2);
            const txt = !Number.isFinite(num) ? "n/a"
              : num === 0 ? "0.000"
              : Math.abs(num) < 0.01 ? num.toExponential(1) : num.toFixed(3);
            return `<td style="background:${bg};text-align:right;font-family:var(--mono);font-size:11px" title="${Number.isFinite(num) ? num : "n/a"}">${txt}</td>`;
          };
          cmEl.innerHTML = `<div style="overflow-x:auto"><table class="blk-matrix">
            <tr><th>channel</th>${kx.map(ci => `<th title="${esc(cm.classes[ci])}">${esc(cm.classes[ci]).slice(0, 8)}</th>`).join("")}</tr>
            ${cm.matrix.slice(0, 30).map((row, ci) =>
              `<tr><th title="${esc(cmap[ci]?.name || ("ch" + (ci + 1)))}">${esc(cmap[ci]?.name || ("ch" + (ci + 1)))}</th>${kx.map(cki => cellv(row[cki])).join("")}</tr>`).join("")}
          </table></div>
          <div class="hint">${esc(cm.note || "")} — green = net excitatory push, red = inhibitory.</div>`;
        }
      }
      // ── WHERE THE FLY BRAIN IS USED (live wiring explainer)
      const wx = $("#br-explainer");
      if (wx) {
        const raws = (S.snap?.formula_values?.[`${S.symbol}|${S.focusTf}`]?.raw) || {};
        const topIn = Object.entries(raws)
          .filter(([fid]) => Number(fid) !== 20)
          .sort((a2, b2) => Math.abs(b2[1]) - Math.abs(a2[1]))
          .slice(0, 3)
          .map(([fid, v]) => `F${fid} ${v >= 0 ? "+" : ""}${fx(v, 3)}`)
          .join(" · ") || "—";
        const dop = fw.dopamine || {};
        const fo = fw.formulas || {};
        wx.innerHTML = `
          <div class="logic" style="line-height:1.7">
            <b>1 · Projection (0–29)</b> — each formula's signed value is written onto its ON/OFF antennal-lobe channels (685 ALPNs, sorted groups). Dominant inputs now: <b>${esc(topIn)}</b><br/>
            <b>2 · Kenyon cells (≈5,177)</b> — sparse ReLU code, top 5 % survive every hop (APL). Active now: <b>${wo?.kc_active ?? "—"}</b><br/>
            <b>3 · Dopamine gates</b> — PAM up with +drg, PPL1 up with −drg; RPE learning only. Now: <b>drg ${fx(dop.drg, 3)} · E ${fx(dop.E, 3)} · α ${fx(dop.alpha, 3)}</b><br/>
            <b>4 · MBON outputs</b> — approach {ACh,GABA} vs avoid {Glu}: <b>${wo ? `${fx(wo.approach, 4)} vs ${fx(wo.avoid, 4)}` : "—"}</b><br/>
            <b>5 · Lateral horn + read-out</b> — CCSv2 <b>${fo.ccsv2?.zero_reasons ? "brain absent" : fx(fo.ccsv2?.value, 4)}</b> · KCAE <b>${fx(fo.kcae?.value, 4)}</b> (gauge) · whole-brain balance <b>${wo ? fx(wo.balance, 4) : "—"}</b>
          </div>
          <div class="hint">fusion: CCI × (1 + ${fw.state === "ready" ? "0.25 × hindsight × vote × base_conf" : "weight 0 — absent"}); the vote's sign is never hand-edited (B-10).</div>`;
      }
      $("#br-fw-note").textContent =
        fw.state === "ready"
          ? `connectome ${fw.meta?.neurons?.toLocaleString?.() ?? "?"} neurons · ${fw.meta?.edges?.toLocaleString?.() ?? "?"} connections · build v${fw.build_version} · vote enters fusion at weight 0.25 × hindsight, never hand-signed`
          : `absent — vote withheld, fusion weight 0, both brain formulas report "brain absent" (${fw.last_error || "not loaded"}).`;
      const rb = $("#br-reconnect");
      if (rb && !rb.dataset.wired) {
        rb.dataset.wired = "1";
        rb.addEventListener("click", async () => {
          rb.disabled = true;
          try {
            const out = await api("/api/brain/reconnect", { method: "POST" });
            toast(out.flywire?.ok !== false ? "reconnect — verify/build restarted" : "reconnect sent", "ok");
          } catch (e) { toast("reconnect: " + e.message, "err"); }
          setTimeout(() => { rb.disabled = false; loadBrains(); }, 1500);
        });
      }
    }
    renderCorel(s.corelation);
    // §3.3 wiring + A-2/A-3 probes (cached server-side after first run)
    try {
      const w = await api(`/api/brain/wiring?symbol=${S.symbol}&tf=${S.focusTf}`);
      const pp = w.push_pull || {};
      $("#br-wiring").innerHTML = `<div class="kv-row">
        <div class="kv"><span>channels oriented</span><b class="${w.all_oriented ? "acc-up" : "acc-dn"}">${(w.channels || []).length - (w.misoriented || 0)}/${(w.channels || []).length}${(w.calibrated || []).length ? ` · calibrated ${w.calibrated.length}` : ""}</b></div>
        <div class="kv"><span>push-pull odd-sum (0 = perfect ✓)</span><b class="${pp.odd_ok ? "acc-up" : "acc-dn"}">${fmt(pp.odd_sum, 4)}${pp.odd_ok ? " ✓" : ""}</b></div>
        <div class="kv"><span>DC offset removed</span><b>${fmt(pp.dc_offset, 4)}</b></div>
        <div class="kv"><span>+1 probe reads</span><b>${pp.oriented ? "approach ✓" : "WRONG VALENCE ✗"}</b></div>
      </div>
      <div class="hint">${esc(pp.note || "")} · raw odd-sum ${fmt(pp.raw_odd_sum, 4)} · ${esc(w.rule || "")}</div>`;
    } catch (e) { console.warn("wiring:", e); }
    const hf = s.hedge || {};
    const em = s.emotion || {};
    const ph = s.physics || {};
    const fvs = S.snap?.formula_values?.[`${S.symbol}|${S.focusTf}`] || {};
    const active = Object.values(fvs.raw || {}).filter(v => Math.abs(v) > 0.7).length;
    $("#wiring-panel").innerHTML = `
      <div class="kv"><span>affinity cells live (|F|&gt;0.7)</span><b>${active}</b></div>
      <div class="kv"><span>B2 experience (EQ)</span><b>${fmt(br.experience, 3)}</b></div>
      <div class="kv"><span>TEDR → P6 DRPT</span><b>${fmt(em.tedr, 4)} → ${fmt(ph.drpt, 2)}</b></div>
      <div class="kv"><span>HF flags IFR/DPAD/SMDI</span><b>${hf.ifr ? "IFR " : ""}${hf.dpad ? "DPAD " : ""}${hf.smdi ? "SMDI " : ""}${!hf.ifr && !hf.dpad && !hf.smdi ? "—" : ""}</b></div>
      <div class="kv"><span>hedge combined</span><b>${fmt(hf.combined, 4)}</b></div>
      <div class="kv"><span>double-check</span><b>${s.double_check ? s.double_check.label + (s.double_check.agree ? " ✓" : " ✗") : "—"}</b></div>
      <div class="hint">Seams update every GRC; corel edges accumulate over the 50-GRC window.</div>`;
  } catch (e) { toast("brains: " + e.message, "err"); }
}
function renderCorel(corel) {
  const cv = $("#corel-canvas"); if (!cv) return;
  const meta = $("#corel-meta");
  const nodes = corel?.nodes || [], edges = corel?.edges || [];
  if (!nodes.length) { meta.textContent = "warming up…"; return; }
  cv.width = cv.clientWidth * 2 || 720; cv.height = 640;
  const g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  const pos = {};
  nodes.forEach((n, i) => {
    const a = (i / nodes.length) * 2 * Math.PI;
    pos[n.id] = { x: cv.width / 2 + Math.cos(a) * cv.width * 0.37,
                  y: cv.height / 2 + Math.sin(a) * cv.height * 0.36 };
  });
  const colors = { brain: "#c9a227", emotion: "#e879f9", physics: "#5b8cff", core: "#00d4f0", agent: "#22c98a" };
  edges.forEach(e => {
    const a = pos[e.a], b = pos[e.b]; if (!a || !b) return;
    g.strokeStyle = (e.corr ?? 0) >= 0 ? "rgba(34,201,138,.7)" : "rgba(240,56,90,.7)";
    g.lineWidth = 1 + Math.abs(e.corr ?? 0) * 5;
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
  });
  nodes.forEach(n => {
    const p = pos[n.id]; if (!p) return;
    g.fillStyle = colors[n.kind] || "#00d4f0";
    g.beginPath(); g.arc(p.x, p.y, 13, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#fff"; g.font = "bold 12px monospace"; g.textAlign = "center";
    g.fillText(n.short || n.id.slice(0, 2), p.x, p.y + 4);
  });
  const mags = edges.map(e => Math.abs(e.corr ?? 0)).filter(v => v > 0);
  meta.textContent = `${nodes.length} nodes · ${edges.length} edges` +
    (mags.length ? ` · avg |r| ${fmt(mags.reduce((a, b) => a + b, 0) / mags.length, 3)}` : " · collecting 50-GRC window");
}

/* ═══════════ AGENTS ═══════════ */
async function loadAgents() {
  try {
    const [d, st] = await Promise.all([
      api("/api/agents/status"),
      api("/api/settings").catch(() => null),
    ]);
    const a = d.agents || {};
    const mode = a.mode || "drosophila";
    const banner = $("#ag-mode");
    banner.className = "mode-banner mode-" + mode;
    banner.innerHTML = mode === "drosophila"
      ? `<b>DROSOPHILA MODE</b> — no keys needed. The 4-instance brain, every core formula, math & physics run everything. Paste any key below for an optional cloud second opinion.`
      : mode === "local" ? `<b>LOCAL MODEL</b> — on-device inference. Cloud keys optional.`
      : `<b>CLOUD AGENTS</b> — ${ (a.slots || []).filter(s => s.status === "connected").length } slot(s) live. Drosophila core still runs every GRC.`;
    // ── local auditor (Ollama) ──
    const aud = a.auditor || {};
    // llama.cpp is the ACTIVE local backend when loaded — the Ollama
    // auditor daemon being down is then optional, never an error
    // (screenshot: red "DAEMON offline" beside a healthy model file)
    const audLocal = !!a.local?.available;
    $("#ag-auditor").innerHTML = `
      <div class="kv-row">
        <div class="kv"><span>daemon</span><b class="${aud.available ? "acc-up" : (audLocal ? "dim" : "acc-dn")}">${aud.available ? "● online" : audLocal ? "○ offline (optional)" : "○ offline"}</b></div>
        <div class="kv"><span>model</span><b>${aud.model || "qwen2.5-coder:3b"}</b></div>
        <div class="kv"><span>triggers (exc / low-acc)</span><b>${aud.triggers?.exceptions ?? 0} / ${aud.triggers?.accuracy ?? 0}</b></div>
        <div class="kv"><span>audits run</span><b>${aud.triggers?.audits ?? 0}</b></div>
      </div>
      ${aud.available ? "" : `<div class="hint">${audLocal ? "optional formula-auditor only — the llama.cpp agent is the active local backend" : (aud.hint || "run: bash scripts/setup_local_llm.sh")}</div>`}
      ${(aud.reports || []).slice(0, 3).map(r => `
        <div class="logic"><b>${r.formula}</b> · ${r.trigger} · ${r.ok ? "✓" : "✗"} ${r.latency_ms}ms
          <br/>${String(r.reply || "").slice(0, 240)}</div>`).join("")}`;

    // ── Groq synthesis status ──
    const gq = a.groq || {};
    $("#ag-groq").innerHTML = gq.configured
      ? `<div class="kv-row">
          <div class="kv"><span>key</span><b class="acc-up">configured</b></div>
          <div class="kv"><span>primary</span><b>${gq.primary}</b></div>
          <div class="kv"><span>news triage</span><b>${gq.triage}</b></div>
        </div>
        <div class="hint">Synthesis runs on mixed double-checks or every 3rd GRC (under the 30 RPM free cap). JSON: regime · weight_adjustments · confidence_score.</div>`
      : `<div class="hint">Set <code>GROQ_API_KEY</code> env var (Codespaces → Settings → Secrets → Codespaces) or import the Groq pack below. Free key: console.groq.com</div>`;

    // ── downloadable agent packs + auto-best scorecard ──
    const cat = a.catalog || {};
    const packs = cat.packs || [];
    $("#ag-packs").innerHTML = `
      <div class="mode-banner ${cat.auto_best ? "local" : ""}">
        ${cat.auto_best
          ? `AUTO-BEST ON — routing to <b>${cat.best || "first available"}</b> (highest realized accuracy)`
          : "AUTO-BEST OFF — fixed slot order (gemini → backup1 → backup2 → backup3)"}
      </div>
      <div style="margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <button class="btn ${cat.auto_best ? "active" : ""}" id="ag-auto-btn">${cat.auto_best ? "disable" : "enable"} auto-best routing</button>
        <span class="dim">auto-best: every agent scores on REAL outcomes; the best one answers first</span>
      </div>
      <div class="key-grid" style="margin-top:12px">
        ${packs.map(pk => `<div class="key-card">
          <b>${pk.name}</b>
          <span>${pk.model} · slot ${pk.slot}</span>
          <div class="dim" style="font-size:11.5px;margin-top:4px">score ${pk.score != null
            ? `${Math.round(pk.score * 100)}% over ${pk.samples} outcomes`
            : (pk.samples ? `${pk.samples} samples` : "no outcomes yet")}${pk.imported ? " · IMPORTED" : ""}</div>
          <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;align-items:center">
            <input type="password" class="key-in" data-key-for="${pk.id}" placeholder="paste ${pk.name} API key" style="flex:1;min-width:130px">
            <button class="btn" data-install="${pk.id}">⬇ install in app</button>
          </div>
          <div class="dim" style="font-size:11px;margin-top:6px"><a href="${pk.setup}" target="_blank" rel="noopener">get API key ↗</a></div>
        </div>`).join("")}
      </div>`;
    const autoBtn = $("#ag-auto-btn");
    if (autoBtn) autoBtn.addEventListener("click", async () => {
      try {
        const r = await fetch("/api/agents/auto", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !cat.auto_best }) });
        const d = await r.json();
        toast(d.auto_best ? `auto-best ON → ${d.best || "drosophila fallback"}` : "auto-best OFF");
        loadAgents();
      } catch (e) { toast("auto toggle failed: " + e.message, "err"); }
    });
    $$("#ag-packs [data-install]").forEach(b => b.addEventListener("click", async () => {
      const key = $(`#ag-packs [data-key-for="${b.dataset.install}"]`)?.value?.trim();
      if (!key) { toast("paste the API key first — use 'get API key ↗' below the card", "err"); return; }
      b.disabled = true; b.textContent = "installing…";
      try {
        const r = await fetch("/api/agents/import", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pack_id: b.dataset.install, key }) });
        const d = await r.json();
        if (!d.ok) throw new Error(d.error || "import failed");
        toast(`${b.dataset.install} agent installed in-app ✓ — press TEST on its slot`);
        loadAgents();
      } catch (e) { toast("install failed: " + e.message, "err"); b.disabled = false; b.textContent = "⬇ install in app"; }
    }));

    const slots = a.slots || [];
    const titles = ["Gemini primary", "Backup 1 · Gemini", "Backup 2 · GPT-4o", "Backup 3 · Claude"];
    $("#ag-slots").innerHTML = slots.map((s, i) => `
      <div class="card">
        <h4><span>${titles[i] || s.status}</span><span class="slot-status ${_slotCls(s.status)}">${esc(s.status)}</span></h4>
        <div class="kv-row">
          <div class="kv"><span>key</span><b>${esc(s.key || "(none)")}</b></div>
          <div class="kv"><span>calls / h</span><b>${s.usage ?? 0} / ${s.limit ?? "—"}</b></div>
        </div>
        ${s.reason ? `<div class="hint acc-warn">last error: ${esc(s.reason)} — press test to re-check</div>` : ""}
        ${s.status === "key stored · untested" ? `<div class="hint">the key is saved; nothing has asked it a question yet. press <b>test</b>.</div>` : ""}
        ${s.last?.insight ? `<div class="logic">“${esc(s.last.insight)}”</div>` : ""}
        <button class="btn test-slot" data-slot="${["gemini", "backup1", "backup2", "backup3"][i]}" style="margin-top:10px">test</button>
      </div>`).join("");
    const lo = a.local || {};
    // `a.installer` is top-level (it used to be null, so this panel had
    // nothing to render); `lo.install` is the same object for older clients
    const ins = a.installer || lo.install || {};
    const installing = ins.state === "installing";
    // Every phase names what is ACTUALLY happening. The old text promised
    // "may compile 10–20 min — normal", which described the failure, not the
    // fix: the ladder now tries a prebuilt wheel, an OPT-IN source build, and
    // Ollama, and the model download is independent of all three.
    const phaseTxt = {
      runtime: "installing the runtime (prebuilt wheel → opt-in source build → Ollama)",
      model: "downloading the model file",
      load: "loading the agent",
    }[ins.phase] || "working";
    const backendTxt = ins.backend
      ? `<span class="acc-up">${esc(ins.backend)}</span>`
      : `<span class="dim">none yet</span>`;
    const stepsHtml = (ins.steps || []).length
      ? `<div class="hint" style="margin-top:6px">runtime ladder:</div>
         <div class="logic" style="font-size:11.5px">${(ins.steps || []).map(st =>
           `${st.ok ? "✓" : "✗"} <b>${esc(st.step)}</b>${st.reason ? " — " + esc(st.reason) : ""}`).join("<br/>")}</div>`
      : "";
    const ollamaHtml = ins.ollama
      ? `<div class="kv"><span>Ollama :11434</span><b class="${ins.ollama.ok ? "" : (lo.available ? "dim" : "")}">${ins.ollama.ok
            ? `up · ${esc((ins.ollama.models || []).join(", ") || "no model pulled")}`
            : `down (${esc(ins.ollama.detail || "?")})${lo.available ? " · optional — llama.cpp active" : ""}`}</b></div>`
      : "";
    const compilerHtml = ins.compiler
      ? `<div class="kv"><span>C/C++ compiler</span><b>${ins.compiler.ok
            ? `present (${esc(ins.compiler.cxx || ins.compiler.cc || "cc")}) — source build needs PREDICT_AGENT_SOURCE_BUILD=1`
            : "absent — a source build could never succeed"}</b></div>`
      : "";
    $("#ag-local").innerHTML = `
      <div class="kv-row">
        <div class="kv"><span>agent</span><b>${lo.available
          ? `<span class="acc-up">${esc(lo.label || "ready")}</span>${lo.backend ? " · " + esc(lo.backend) : ""}`
          : `<span class="acc-warn">not loaded</span>`}</b></div>
        <div class="kv"><span>runtime</span><b>${backendTxt}</b></div>
        <div class="kv"><span>RAM tier</span><b>${lo.ram_gb ?? "?"} GB · ${esc(ins.model_label || "—")}</b></div>
        <div class="kv"><span>last inference</span><b>${lo.last_ms ? lo.last_ms + " ms" : "—"}</b></div>
      </div>
      <div class="kv-row">
        <div class="kv"><span>model file</span><b>${ins.model_present
          ? `✓ ${esc((ins.validation || {}).sha || "present")}`
          : `${(ins.bytes_on_disk || 0).toLocaleString()} B on disk (${esc(ins.model_size || "?")} expected)`}</b></div>
        ${ollamaHtml}
        ${compilerHtml}
      </div>
      ${(() => {
        // ── THE NO-DOWNLOAD ROUTES ────────────────────────────────────
        // When huggingface.co is blocked the GGUF button can never work, and
        // pressing it again is the only thing the old panel offered. Show
        // what CAN work instead, with the exact setting for each.
        const lep = a.local_endpoint || {};
        const alt = lep.alternatives || {};
        const live = lep.ok
          ? `<div class="kv"><span class="acc-up">ALREADY RUNNING</span>`
            + `<b>${esc(lep.label || "local server")} · ${esc(lep.base || "")}`
            + `${lep.model ? " · " + esc(lep.model) : ""}</b></div>`
            + `<div class="hint">no download needed — the app talks to it `
            + `over the OpenAI-compatible API</div>`
          : `<div class="kv"><span class="acc-warn">no local server found</span>`
            + `<b class="dim">${esc((lep.tried || []).length)} port(s) probed: `
            + `${esc((lep.known_servers || []).slice(0, 4).join(", "))}…</b></div>`;
        const starters = (alt.start_a_server || []).map(x =>
          `<div class="kv"><span>${esc(x.what)}</span>`
          + `<b class="dim">${esc(x.then)} · ${esc(x.url)}</b></div>`).join("");
        const cloud = (alt.free_cloud || []).map(x =>
          `<div class="kv"><span>${esc(x.what)}</span>`
          + `<b class="dim">${esc(x.setting)} · ${esc(x.model)} · ${esc(x.url)}</b></div>`).join("");
        const manual = alt.manual_gguf || {};
        return `<div class="hint" style="margin-top:10px"><b>Works without any download</b></div>`
          + live + starters
          + `<div class="hint" style="margin-top:6px"><b>or a free cloud key</b></div>` + cloud
          + (manual.path ? `<div class="kv"><span>or drop a .gguf</span><b class="dim">`
             + `download it in a browser on any machine, then place it at ${esc(manual.path)}</b></div>` : "")
          + `<div class="kv-row" style="margin-top:8px">`
          + `<input class="inp" id="ag-endpoint" placeholder="http://host:port" `
          + `value="${esc(lep.base || "")}" style="flex:1"/>`
          + `<button class="btn" id="ag-endpoint-set">use this server</button></div>`;
      })()}
      ${stepsHtml}
      ${ins.installed
        ? `${lo.available
            ? `<div class="hint" style="margin-top:8px">✓ local agent ready and loaded</div>`
            : `<div class="hint acc-warn" style="margin-top:8px">⚠ the runtime is INSTALLED but NOT answering${lo.error ? ` — ${esc(lo.error)}` : ""}. Nothing is being asked locally; every analysis runs on the Drosophila brain + formulas. Press re-probe.</div>`}
           <button class="btn test-slot" data-slot="local" style="margin-top:10px">test</button>
           <button class="btn" id="ag-retry" style="margin-top:10px">re-probe runtime</button>`
        : installing
        ? `<div class="hint" style="margin-top:8px">⬇ ${phaseTxt}… ${ins.elapsed_s ? Math.floor(ins.elapsed_s / 60) + "m " + (ins.elapsed_s % 60) + "s" : ""} — the app keeps running; this panel refreshes itself</div>
           <div class="logic" style="max-height:110px;overflow:auto;font-size:11.5px">${(ins.log || []).slice(-6).map(esc).join("<br/>")}</div>
           <button class="btn" disabled style="margin-top:10px">downloading…</button>`
        : `<button class="btn" id="ag-download" style="margin-top:10px">⬇ DOWNLOAD LOCAL AGENT</button>
           <button class="btn" id="ag-retry" style="margin-top:10px">re-probe runtime</button>
           <div class="hint" style="margin-top:6px">The runtime is a ladder: a prebuilt wheel first, a source build only if you allow it, then Ollama if it is installed — nothing here can block the app. The model file (${esc(ins.model_label || "preferred")}, ${esc(ins.model_size || "")}) downloads independently of it, so a host with no compiler still banks the model.</div>
           ${(() => {
             const au = ins.auto || {};
             const next = au.next_try_in_s;
             if (au.state === "running")
               return `<div class="hint" style="margin-top:6px">⬇ auto-download is armed — ${next != null && next > 0 ? "next attempt in " + Math.floor(next / 60) + "m " + (next % 60) + "s" : "attempting now"}${au.attempts ? " · attempt " + au.attempts + " already failed" : ""}${au.last_error ? " · " + esc(au.last_error) : ""}</div>`;
             if (au.state === "failed")
               return `<div class="hint" style="margin-top:6px">⚠ auto-download gave up after ${au.attempts || 0} attempt(s): ${esc(au.last_error || au.last_result || "unknown")} — press download to retry now (it also retries on the next boot)</div>`;
             return `<div class="hint" style="margin-top:6px">auto-download: ${esc(au.state || "idle")} — the engine fetches this by itself after boot; the button is only for "now"</div>`;
           })()}
           ${ins.state === "failed" ? `<div class="hint">⚠ last attempt stopped: ${esc(ins.error || ins.why || "unknown")} — press download to retry/resume</div>` : ""}
           ${lo.error ? `<div class="hint">${esc(lo.error)}</div>` : ""}`}
      `;
    const dlBtn = $("#ag-download");
    if (dlBtn) dlBtn.addEventListener("click", async () => {
      dlBtn.disabled = true; dlBtn.textContent = "starting…";
      try {
        const r = await api("/api/agents/local/install", { method: "POST" });
        toast(r.install?.state === "done" ? "local agent already downloaded"
                                           : "local agent download started — watch this panel");
      } catch (e) { toast("download start failed: " + e.message, "err"); }
      loadAgents();
    });
    const epBtn = $("#ag-endpoint-set");
    if (epBtn) epBtn.addEventListener("click", async () => {
      epBtn.disabled = true; epBtn.textContent = "probing…";
      try {
        const ep = ($("#ag-endpoint") || {}).value || "";
        const r = await api("/api/agents/local/endpoints",
                            { method: "POST", body: JSON.stringify({ endpoint: ep }) });
        toast(r.ok && (r.local || {}).available
          ? "connected to " + (r.local.endpoint || ep)
          : "no server answered at " + (ep || "(autodetect)"), 
          r.local && r.local.available ? "ok" : "err");
      } catch (e) { toast("endpoint probe failed: " + e.message, "err"); }
      loadAgents();
    });
    const retryBtn = $("#ag-retry");
    if (retryBtn) retryBtn.addEventListener("click", async () => {
      retryBtn.disabled = true; retryBtn.textContent = "probing…";
      try {
        const r = await api("/api/agents/local/retry", { method: "POST" });
        toast(r.ok ? `runtime found: ${r.backend || "local"}` : `still unavailable — ${r.error || "see the ladder below"}`,
              r.ok ? "" : "warn");
      } catch (e) { toast("re-probe failed: " + e.message, "err"); }
      loadAgents();
    });
    const autoRunning = (ins.auto || {}).state === "running";
    if ((installing || autoRunning) && S.view === "agents") {
      clearTimeout(window.__agPoll);
      window.__agPoll = setTimeout(() => { if (S.view === "agents") loadAgents(); }, 4000);
    }
    const sig = d.signal || {};
    $("#ag-signal").innerHTML = sig.source && sig.source !== "none"
      ? `<div class="kv-row">
          <div class="kv"><span>direction</span><b>${sig.direction > 0 ? "▲ confirm" : sig.direction < 0 ? "▼ contradict" : "0 uncertain"}</b></div>
          <div class="kv"><span>confidence</span><b>${fmt(sig.confidence, 3)}</b></div>
          <div class="kv"><span>weight</span><b>${fmt(sig.weight, 4)}</b></div>
          <div class="kv"><span>source</span><b>${sig.source}${sig.fallback ? " (fb)" : ""}</b></div>
        </div>
        ${sig.insight ? `<div class="logic">“${esc(sig.insight)}”</div>` : ""}`
      : `<div class="hint">No cloud signal — fine. Drosophila brain + formulas carry every vote; the 4-vote check still runs.</div>`;
    const map = { "key-gemini": "gemini_key", "key-backup1": "backup1_key", "key-backup2": "backup2_key",
                  "key-backup3": "backup3_key", "key-newsapi": "newsapi_key", "key-neuprint": "neuprint_key" };
    const sset = st?.settings;
    // NEVER fill a password field with the key (or its mask). A stored key is
    // announced by the placeholder; an empty submit means "keep it".
    if (sset) for (const [id, k] of Object.entries(map)) {
      const el = $("#" + id);
      if (!el) continue;
      el.value = "";
      el.placeholder = sset[k] ? "stored — paste a new key to replace" : "paste API key";
    }
    $$(".test-slot").forEach(b => b.addEventListener("click", async () => {
      b.textContent = "…";
      try {
        const r = await api(`/api/agents/test/${b.dataset.slot}`, { method: "POST" });
        toast(`${b.dataset.slot}: ${r.result?.ok ? "OK" : "FAILED — " + (r.result?.detail?.reason || r.result?.error || "")}`,
          r.result?.ok ? "" : "warn");
      } catch (e) { toast(e.message, "err"); }
      b.textContent = "test"; loadAgents();
    }));
  } catch (e) { toast("agents: " + e.message, "err"); }
}


/* ═══════════ MIND — cognition + metacognition ═══════════ */
/* The thinking trace is produced by app/cognition.py. This view only draws
   what the server reasoned; it never composes a step of its own, because a
   UI that invents a reason is worse than one that shows none. */
let _mindLast = null;
async function loadMind(force) {
  if (!force && _mindLast && Date.now() - _mindLast < 20000) return;
  _mindLast = Date.now();
  const sc = $("#mind-scope");
  if (sc) sc.textContent = `${S.symbol} · ${S.focusTf.toUpperCase()}`;
  try {
    const t = await api(`/api/think?symbol=${S.symbol}&tf=${S.focusTf}`);
    renderMind(t);
  } catch (e) {
    const box = $("#mind-steps");
    if (box) box.innerHTML = `<div class="dim pad">trace unavailable: ${esc(e.message)}</div>`;
  }
  try { renderMindHistory(await api(`/api/think/history?symbol=${S.symbol}&tf=${S.focusTf}`)); }
  catch (e) { /* ledger is optional */ }
}

const MIND_STATUS = { done: "acc-up", partial: "dim", warn: "acc-warn",
                      conflict: "acc-warn" };

function renderMind(t) {
  // ── THE SPINE: the same six steps as a compact, coloured chain.
  // The prose below is honest but reads as a wall of text ("mind is just
  // text"), so the state of the whole reasoning is also drawn as one row
  // you can scan: each block is a step, its colour is its status. Still
  // server-driven — a step the server did not send has no block here.
  const spine = $("#mind-spine");
  if (spine) {
    const st = t.steps || [];
    const doneN = st.filter(s => s.status === "done").length;
    const badN = st.filter(s => s.status === "warn" || s.status === "conflict").length;
    spine.innerHTML = `
      <div class="mind-chain">${st.map((s, i) => `
        <span class="mind-node ${esc(s.status)}" title="${esc(s.title)} — ${esc(s.status)}">
          <i>${i + 1}</i><b>${esc((s.step || "").slice(0, 4).toUpperCase())}</b>
        </span>${i < st.length - 1 ? '<span class="mind-link"></span>' : ""}`).join("")}
      </div>
      <div class="dim" style="font-size:11px;margin-top:6px">
        ${doneN}/${st.length} steps settled${badN ? ` · <span class="acc-warn">${badN} caveat${badN > 1 ? "s" : ""}</span>` : " · no caveats"}
        ${t.consistency ? (t.consistency.ok
            ? ' · <span class="acc-up">self-consistent</span>'
            : ' · <span class="acc-dn">CONTRADICTION</span>') : ""}
      </div>`;
  }

  // ── the six steps, verbatim from the server
  const box = $("#mind-steps");
  if (box) {
    box.innerHTML = (t.steps || []).map((s, i) => `
      <div class="pad" style="border-left:2px solid var(--${s.status === "warn" || s.status === "conflict" ? "gold" : "line"});margin:6px 0;padding-left:10px">
        <div style="display:flex;gap:8px;align-items:baseline;flex-wrap:wrap">
          <b class="dim" style="font-family:var(--mono);font-size:11.5px">${i + 1}. ${esc(s.step.toUpperCase())}</b>
          <b>${esc(s.title)}</b>
          <span class="${MIND_STATUS[s.status] || "dim"}" style="font-family:var(--mono);font-size:11px">${esc(s.status)}</span>
        </div>
        <div class="logic" style="font-size:11.5px">${esc(s.detail)}</div>
        ${(s.cites || []).length ? `<div class="hint" style="font-size:11px">source: ${s.cites.map(esc).join(" · ")}</div>` : ""}
      </div>`).join("")
      || `<div class="dim pad">the server returned no steps</div>`;
  }

  // ── metacognition: what it thinks of itself, and whether that is earned
  const c = t.competence || {};
  const ab = t.abstain || {};
  const meta = $("#mind-meta");
  if (meta) {
    const trustCls = c.trust === "calibrated" ? "acc-up"
      : c.trust === "overconfident" ? "acc-dn"
      : c.trust === "underconfident" ? "acc-warn" : "dim";
    meta.innerHTML = `
      <div class="kv"><span>trust in itself</span><b class="${trustCls}">${esc(c.trust || "unknown")}</b></div>
      <div class="kv"><span>graded outcomes</span><b>${c.n ?? 0}${c.state ? ` · ${esc(c.state)}` : ""}${!(c.n)
        ? ` <span class="dim" style="font-size:11px">${esc(S.snap?.ledger?.why_empty || "the ledger is written at the first window boundary")}</span>` : ""}</b></div>
      <div class="kv"><span>hit-rate</span><b>${c.hit_rate != null ? (c.hit_rate * 100).toFixed(0) + "%" : "—"}
        ${c.wilson_lo != null ? `<span class="dim" style="font-size:11px">95% floor ${(c.wilson_lo * 100).toFixed(0)}%</span>` : ""}</b></div>
      <div class="kv"><span>Brier</span><b>${c.brier != null
        ? `${fmt(c.brier, 3)} <span class="dim" style="font-size:11px">rel ${fmt(c.brier_parts?.reliability, 3)} · res ${fmt(c.brier_parts?.resolution, 3)}</span>`
        : `<span class="dim">${esc(c.brier_parts?.why || "not enough settled calls")}</span>`}</b></div>
      <div class="kv"><span>verdict</span><b class="${ab.abstain ? "acc-warn" : "dim"}">${ab.abstain ? "ABSTAINING" : "proceeding"}</b></div>`;
  }
  const hon = $("#mind-honesty");
  if (hon) hon.textContent = c.how_sure || "";
  if (hon && ab.reason) hon.textContent += (hon.textContent ? " · " : "") + ab.reason;

  const cf = $("#mind-cf");
  if (cf) cf.textContent = t.counterfactual || "—";

  // ── working memory: present vs withheld, each with its reason
  const wm = t.working_memory || {};
  const wmBox = $("#mind-wm");
  if (wmBox) {
    const pill = (x, ok) => `<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border-radius:9px;font-size:11.5px;border:1px solid var(--line);color:var(--${ok ? "green" : "gold"})">${esc(x.label)}</span>`;
    wmBox.innerHTML = `
      <div class="dim" style="font-size:11px">${esc(wm.summary || "")}</div>
      <div style="margin-top:6px">${(wm.present || []).map(p => pill(p, true)).join("") || '<span class="dim">nothing reached the reasoning</span>'}</div>
      ${(wm.withheld || []).length ? `<div class="panel-h" style="margin-top:8px">WITHHELD — not silence, a stated abstention</div>
        ${wm.withheld.map(w => `<div class="pad" style="font-size:11px"><b class="acc-warn">${esc(w.label)}</b> <span class="dim">— ${esc(w.why || "")}</span></div>`).join("")}` : ""}`;
  }

  // ── assumptions: a premise is a claim, so it gets a verdict
  const asBox = $("#mind-assume");
  if (asBox) {
    const rows = (t.assumptions || []).map(a => {
      const tag = a.holds === null ? `<span class="dim">unverifiable</span>`
        : a.holds ? `<span class="acc-up">holds</span>`
                  : `<span class="acc-dn">DOES NOT HOLD</span>`;
      return `<div class="pad" style="font-size:11px;border-left:2px solid var(--${a.holds === false ? "red" : "line"});padding-left:8px;margin:4px 0">
        <div>${tag} — <b>${esc(a.claim)}</b></div>
        <div class="dim" style="font-size:11.5px">${esc(a.evidence || "")}</div></div>`;
    }).join("");
    asBox.innerHTML = rows || `<div class="dim">no assumptions reported</div>`;
  }

  // ── self-check
  const chk = $("#mind-check");
  if (chk && t.consistency) {
    chk.innerHTML = t.consistency.ok
      ? `<div class="kv"><span>internal consistency</span><b class="acc-up">holds</b></div>
         <div class="kv"><span>checked</span><b class="dim">${(t.consistency.checked || []).join(" · ")}</b></div>`
      : `<div class="kv"><span>internal consistency</span><b class="acc-dn">CONTRADICTION</b></div>`
        + (t.consistency.problems || []).map(p => `<div class="hint acc-dn">${esc(p)}</div>`).join("");
  }
}

function renderMindHistory(h) {
  const box = $("#mind-history");
  if (!box) return;
  const recs = (h.records || []).slice().reverse();
  const st = h.thought_stats || {};
  box.innerHTML = `
    <div class="kv-row">
      <div class="kv"><span>traces recorded</span><b>${h.n ?? 0}</b></div>
      <div class="kv"><span>graded by settlement</span><b>${h.graded ?? 0}${h.graded ? ` · ${h.wins} won` : ""}</b></div>
      <div class="kv"><span>reasoning hit-rate</span><b>${st.hit_rate != null ? (st.hit_rate * 100).toFixed(0) + "%" : "—"}${st.n ? ` <span class="dim">n=${st.n}</span>` : ""}</b></div>
      <div class="kv"><span>open / abstained</span><b>${st.open ?? 0} / ${st.abstained ?? 0}</b></div>
      <div class="kv"><span>scope</span><b class="dim">${esc(st.scope || "all")}</b></div>
      <div class="kv"><span>trust</span><b class="dim">${esc(st.trust || "unknown")}</b></div>
    </div>
    <div style="font-size:11.5px;font-family:var(--mono);margin-top:6px">
      ${recs.map(r => `<div style="display:flex;gap:8px;flex-wrap:wrap;color:var(--dim)">
        <span>${new Date(r.ts * 1000).toISOString().slice(5, 16).replace("T", " ")}</span>
        <span>${esc(r.signal || "—")}</span>
        <span>conf ${fmt(r.confidence, 3)}</span>
        <span>${esc(r.signature || "")}</span>
        ${r.lock_id ? `<span class="dim">🔒 ${esc(String(r.lock_id).slice(-8))}</span>` : ""}
        <span class="${r.outcome === "win" ? "acc-up" : r.outcome === "loss" ? "acc-dn" : "dim"}">${r.outcome || "open"}</span>
        ${r.abstain ? `<span class="acc-warn">abstained</span>` : ""}
        ${r.split ? `<span class="acc-warn">routes split</span>` : ""}
        ${r.consistency === false ? `<span class="acc-dn">trace contradicted itself</span>` : ""}
      </div>`).join("") || '<span class="dim">no traces recorded yet — run ▶ ANALYSE</span>'}
    </div>
    <div class="hint" style="margin-top:6px">${esc(h.rule || "")}</div>`;
}

/* ═══════════ HISTORY ═══════════ */
/* Self-examination of the sealed prediction journal. The app must be able
   to say "one of my own records is incoherent", name it, and keep going. */
async function fetchPredictionAudit() {
  const box = $("#pred-audit");
  if (!box) return null;
  try {
    const a = await api("/api/predictions/audit");
    if (a.error) {
      box.className = "hint";
      box.innerHTML = `<span class="acc-warn">⚠ SELF-AUDIT FAILED — ${esc(a.error)}</span>`;
      return a;
    }
    const issues = a.issues || [];
    if (a.ok && !issues.length) {
      box.className = "hint";
      box.innerHTML = `<span class="dim">✓ self-audit: ${a.checked ?? 0} sealed row(s) coherent — no wrong-side ladder, no settlement younger than one bar.</span>`;
      return a;
    }
    box.className = "hint";
    box.innerHTML = `<span class="acc-warn">⚠ SELF-AUDIT FOUND ${issues.length} INCONSISTENT ROW(S)</span>` +
      `<div class="dim" style="margin-top:4px">${esc(a.rule || "")}</div>` +
      `<ul style="margin:6px 0 0 16px">${issues.slice(0, 8).map(i =>
        `<li><span class="mono">${esc(i.symbol || "?")}·${esc(String(i.tf || "").toUpperCase())}</span> ` +
        `<span class="acc-warn">${esc(i.kind || "issue")}</span> — ${esc(i.detail || "")}</li>`).join("")}` +
      `${issues.length > 8 ? `<li class="dim">…and ${issues.length - 8} more</li>` : ""}</ul>`;
    return a;
  } catch (e) {
    box.className = "hint";
    box.innerHTML = `<span class="acc-warn">⚠ self-audit unavailable: ${esc(e.message)}</span>`;
    return null;
  }
}

async function loadSealedPredictions() {
  const box = $("#pred-history");
  if (!box) return;
  loadDemoSync().catch(() => {});
  // SELF-AUDIT (live defect 2026-11-27): the sealed journal checks its own
  // rows against its own invariants. A clean report is a quiet line; a
  // broken one is an error the page shows instead of a number the user has
  // to notice for us. Fetched separately so a failure here can never hide
  // the history table itself.
  fetchPredictionAudit().catch(() => {});
  try {
    const [p, s] = await Promise.all([
      api("/api/predictions?limit=40"),
      api("/api/predictions/stats"),
    ]);
    const st = p.stats || s || {};
    const meta = $("#pred-meta");
    if (meta) meta.textContent =
      `${st.graded ?? 0} settled · hit ${st.hit_rate != null ? Math.round(st.hit_rate * 100) + "%" : "—"}` +
      ` · avg ${st.avg_bps != null ? (st.avg_bps >= 0 ? "+" : "") + fmt(st.avg_bps, 1) + " bp" : "—"}` +
      ` · ${st.open ?? 0} open`;
    const rows = (p.locks || []).map(r => {
      const outcome = r.outcome === "won" ? '<span class="acc-up">WON</span>'
        : r.outcome === "lost" ? '<span class="acc-dn">LOST</span>'
        : r.outcome === "push" ? '<span class="dim">PUSH</span>'
        : r.outcome === "ambiguous" ? '<span class="dim" title="Both TP and SL were inside one OHLC candle; order is unknowable, excluded from accuracy.">AMBIGUOUS</span>'
        : '<span class="dim">open</span>';
      const when = Number(r.created_at) ? new Date(r.created_at * 1000).toISOString().slice(5, 16).replace("T", " ") : "—";
      return `<tr>
        <td class="mono">${esc(r.symbol)}</td>
        <td class="mono">${esc(String(r.tf || "").toUpperCase())}</td>
        <td class="mono">${when}</td>
        <td class="mono ${r.side === "BUY" ? "acc-up" : "acc-dn"}">${esc(r.side || "—")}</td>
        <td class="mono">${px(r.entry)}</td>
        <td class="mono">${px(r.tp)}</td>
        <td class="mono">${px(r.sl)}</td>
        <td class="mono">${fmt(r.confidence, 3)}</td>
        <td class="mono">${r.kind === "watch" ? "watch" : "issued"}</td>
        <td>${outcome}${r.bps != null ? ` <span class="mono dim">${Number(r.bps) >= 0 ? "+" : ""}${fmt(r.bps, 1)}bp</span>` : ""}</td>
        <td class="dim" style="text-transform:none">${esc(r.settle_reason || (r.why || []).join("; ") || r.reason || "")}</td>
      </tr>`;
    }).join("");
    box.innerHTML = `<table class="tbl"><thead><tr>
      <th>ASSET</th><th>TF</th><th>LOCKED</th><th>SIDE</th><th>ENTRY</th>
      <th>TP</th><th>SL</th><th>CONF</th><th>KIND</th><th>OUTCOME</th><th>WHY</th>
      </tr></thead><tbody>${rows || '<tr><td colspan="11" class="dim">no sealed predictions yet — press ▶ ANALYSE</td></tr>'}</tbody></table>`;
  } catch (e) {
    box.innerHTML = `<div class="dim pad">sealed predictions unavailable: ${esc(e.message)}</div>`;
  }
}

async function loadAnalyses() {
  const box = $("#an-journal"); if (!box) return;
  try {
    const j = await api("/api/analyses?limit=60");
    const rows = (j.records || []).slice().reverse();
    $("#an-j-meta").textContent = `${j.total ?? rows.length} run(s) logged`;
    if (!rows.length) {
      box.innerHTML = `<div class="dim pad">no analyses yet — press ▶ ANALYSE; every run lands here.</div>`;
      return;
    }
    box.innerHTML = `<table class="tbl"><thead><tr>
      <th>WHEN</th><th>SYM/TF</th><th>SIGNAL</th><th>CONF</th><th>VERDICT</th>
      <th>LOCK</th><th>RECHECK</th><th>MS</th><th>LEARNED / NEEDS CHANGE</th>
      </tr></thead><tbody>${rows.map(r => {
        const when = r.started_at ? new Date(r.started_at * 1000) : new Date(r.ts * 1000);
        const sig = r.signal || "—";
        const sigCls = sig === "BUY" ? "acc-up" : sig === "SELL" ? "acc-dn" : "dim";
        const learn = r.ai_correction
          ? `⚠ ${esc(r.ai_correction.by || "AI")} → ${r.ai_correction.direction > 0 ? "BUY" : "SELL"}: ${esc(r.ai_correction.reason || "")}`
          : r.hindsight?.note ? `hindsight: ${esc(String(r.hindsight.note).slice(0, 140))}`
          : r.status === "error" ? `error: ${esc((r.error || "").slice(0, 140))}`
          : r.status === "duplicate" ? "duplicate — followed the running job"
          : r.status === "queued" ? "queued behind another run"
          : `<span class="dim">—</span>`;
        return `<tr>
          <td class="mono">${when.toISOString().slice(5, 16).replace("T", " ")}</td>
          <td>${esc(r.symbol || "?")}/${esc(String(r.tf || "?").toUpperCase())}</td>
          <td class="${sigCls}">${esc(sig)}</td>
          <td class="mono">${r.confidence != null ? fx(r.confidence, 3) : "—"}</td>
          <td>${esc(r.verdict || "—")}${r.gate_ok ? " ✓" : ""}</td>
          <td class="mono">${r.locked ? `🔒 ${esc(String(r.lock_id || "").slice(-8))}` : `<span class="dim">no</span>`}</td>
          <td>${r.recheck_passed ? `<span class="acc-up">pass</span>` : r.recheck ? `<span class="acc-dn">fail</span>` : `<span class="dim">—</span>`}</td>
          <td class="mono">${r.elapsed_ms != null ? Math.round(r.elapsed_ms) : "—"}</td>
          <td class="dim" style="text-transform:none;max-width:340px">${learn}</td>
        </tr>`;
      }).join("")}</tbody></table>`;
  } catch (e) {
    box.innerHTML = `<div class="dim pad">journal unavailable: ${esc(e.message)} — <a href="#" onclick="loadAnalyses();return false">retry</a></div>`;
  }
}

async function loadHistory() {
  try {
    loadSealedPredictions().catch(() => {});
    loadAnalyses().catch(() => {});
    // 24h warm-history seed status — the banner explains WHERE the
    // history below came from (real replayed tape vs live windows)
    try {
      const sd = await api("/api/history/seed");
      const s = sd.seed || {};
      const el = $("#seed-banner");
      if (el) {
        if (s.state === "done") {
          el.innerHTML = `<div class="pad"><b>🌱 24H WARM-HISTORY SEED</b> — replayed the real backfilled 1m tape through the app's own formulas: ` +
            `<b>${s.settled ?? 0} windows settled</b> · <b>${s.graded ?? 0} graded</b> into the ledger, formula leaderboard, calibration rings and confidence pools ` +
            `<span class="dim">(formulas-only replay — brains / book-physics / news need live state and are never faked; seeded ledger rows are flagged)</span></div>`;
        } else if (s.state === "seeding" || s.state === "pending") {
          el.innerHTML = `<div class="pad"><b>🌱 24H WARM-HISTORY SEED</b> — <span class="dim">replaying the last 24h of tape in the background… reload in a moment</span></div>`;
        } else if (s.state === "skipped-sim") {
          el.innerHTML = `<div class="pad"><b>🌱 24H WARM-HISTORY SEED</b> — <span class="acc-warn">skipped: feed is simulated — sim history dressed as a backtest would be a lie. Connect Binance for the real seed.</span></div>`;
        } else if (["partial-history", "insufficient-history", "persistence-failed", "unavailable"].includes(s.state)) {
          const coverage = Object.values(s.coverage || {});
          const bars = coverage.reduce((n, item) => n + (item.requested_bars || 0), 0);
          const expected = coverage.reduce((n, item) => n + (item.expected_requested_bars || 0), 0);
          const reason = s.why || "verified history is not sufficient for a complete replay";
          el.innerHTML = `<div class="pad"><b>🌱 24H WARM-HISTORY SEED</b> — <span class="acc-warn">${esc(reason)}</span>` +
            `${expected ? ` <span class="dim">verified minute coverage: ${bars}/${expected} requested across assets; missing bars were not filled.</span>` : ""}</div>`;
        } else if (s.state === "done-sim" || s.state === "fresh-sim" || s.state === "done") {
          const subs = Object.entries(s.subsystems || {})
            .map(([k, v]) => `${k} ${v.acc == null ? "—" : Math.round(v.acc * 100) + "%"} (n=${v.n})`)
            .join(" · ");
          el.innerHTML = `<div class="pad"><b>🌱 WARM-HISTORY REPLAY</b> — walked the backfilled ${Math.round((s.hours || 168) / 24)}-day tape forward through the app's own formulas, brains, emotions, physics, hedge-flow and advanced layers: ` +
            `<b>${s.new_settled ?? s.settled ?? 0} windows settled</b> · <b>${s.graded ?? 0} graded</b>` +
            `${s.duplicates ? ` · ${s.duplicates} already present (idempotent)` : ""}` +
            `<div class="dim" style="margin-top:4px">${esc(s.why || "")}${subs ? ` · subsystem hit-rates: ${subs}` : ""} · seeded rows are flagged “warm-v2” and can be excluded from any export.</div></div>`;
        } else if (s.state === "fresh") {
          el.innerHTML = `<div class="pad"><b>🌱 24H WARM-HISTORY SEED</b> — <span class="dim">already seeded in the last 24h (${s.rows ?? sd.ledger_rows ?? 0} ledger rows) — history below includes the replayed tape, flagged per row.</span></div>`;
        } else if (s.state === "failed") {
          el.innerHTML = `<div class="pad"><b>🌱 24H WARM-HISTORY SEED</b> — <span class="acc-warn">seed failed this boot — history fills from live windows only; check server log.</span></div>`;
        }
      }
    } catch { /* banner optional */ }
    // advanced module scoreboard (per-module walk-forward accuracy)
    try {
      // BUG FIX: this block referenced an undefined `state` object, so the
      // request URL threw a ReferenceError which the outer catch swallowed —
      // the panel kept its static "Loading…" forever. It now uses the live
      // selection and renders a plain-English state for thin/absent data.
      const advSym = S.symbol, advTf = S.focusTf;
      const adv = await api(`/api/history/advanced?symbol=${advSym}&tf=${advTf}`);
      const mods = adv.modules || {};
      const names = Object.keys(mods);
      const elA = $("#hs-advanced");
      if (elA) {
        const head = `<div class="hint" style="margin:0 0 6px">${advSym} · ${advTf.toUpperCase()} (${TF_META[advTf]?.mins ?? "?"}m) — follows the asset/timeframe selected in the toolbar.</div>`;
        if (!names.length) {
          elA.innerHTML = head + `<div class="dim pad">No module history for ${advSym} ${advTf.toUpperCase()} yet — each module is graded one bar after a lock settles.</div>`;
        } else {
          elA.innerHTML = head + names.map(m => { const d = mods[m] || {};
            const nn = Number(d.n || 0);
            const acc = Number.isFinite(Number(d.acc)) && d.acc != null ? Number(d.acc) : null;
            const enough = nn >= 8 && acc != null;
            return `<div class="kv"><span>${m}</span>` + (enough
              ? `<b class="${acc >= 0.55 ? "acc-up" : acc <= 0.45 ? "acc-dn" : ""}">${(acc * 100).toFixed(1)}% <span class="dim" style="font-weight:400">n=${nn} · weight ×${d.scale ?? 1}</span></b>`
              : `<b class="dim">— n=${nn}/${8} · ${nn === 0 ? "no graded sample yet" : "collecting"}${d.scale != null && Number(d.scale) < 0 ? ` · voting flipped ×${d.scale}` : ""}</b>`) + `</div>`; }).join("")
            + `<div class="hint">Walk-forward graded: module sign locked one pass, settled against the realised bar. Weight scale is SIGNED and learned per asset+TF: acc 50% → ×+0.5, 65% → ×+1.5, below 50% the module votes FLIPPED (×−0.5…−1.5); arms at n≥40. Path-signature: ${adv.signature_samples || 0}/500 graded → ${adv.signature_floor ? "<b>floor weight</b> (unproven)" : "<b>earned weight</b>"}${adv.conformal && adv.conformal.q_bps != null ? ` · conformal q ${adv.conformal.q_bps} bps (n=${adv.conformal.n})` : ""}</div>`;
        }
      }
    } catch { /* scoreboard optional */ }
    // issue 7: each panel gets its OWN try/catch — one failed request
    // used to throw out of the shared Promise.all and leave the day
    // table + leaderboard stuck on "loading…" forever
    let dd = [], lb = {};
    try {
      const days = await api("/api/history/days");
      dd = days.days || [];
      const box = $("#hs-days");
      box.innerHTML = `<div class="day-row head"><span>DATE</span><span>PROGRESS</span><span>GRC</span><span>PRED</span><span>HITS</span></div>` +
        dd.map(d => `
          <div class="day-row" data-date="${d.date}">
            <span>${d.date}</span>
            <span class="bar"><i style="width:${clamp((d.grcs / 360) * 100, 2, 100)}%"></i></span>
            <span>${d.grcs}</span><span>${d.predictions}</span>
            <span>${d.targets_hit ?? 0}/${d.sls_hit ?? 0}</span>
          </div>`).join("") || `<div class="dim pad">no days yet</div>`;
      $$(".day-row[data-date]").forEach(r => r.addEventListener("click", () => showDay(r.dataset.date)));
    } catch (e) {
      const box = $("#hs-days");
      if (box) box.innerHTML = `<div class="dim pad">day table unavailable: ${esc(e.message)} — <a href="#" onclick="loadHistory();return false">retry</a></div>`;
    }
    try {
      lb = await api("/api/history/leaderboard");
      $("#hs-leader").innerHTML = (lb.leaderboard || []).map(r => {
        const nn = r.n ?? 0;
        const enough = nn >= 8;
        return `<div class="kv"><span title="formula F${r.formula ?? ""}">${esc(r.name || r.symbol || r.tf || ("F" + r.formula))}</span>` +
          (enough
            ? `<b class="${(r.accuracy ?? 0) >= 0.55 ? "acc-up" : (r.accuracy ?? 0) <= 0.45 ? "acc-dn" : ""}">${fx(r.accuracy ?? r.score, 4)} <span class="dim" style="font-weight:400">n=${nn}</span></b>`
            : `<b class="dim">— n=${nn} (needs ≥8)</b>`) + `</div>`;
      }).join("")
        || `<div class="dim pad">no resolved predictions yet</div>`;
    } catch (e) {
      const el = $("#hs-leader");
      if (el) el.innerHTML = `<div class="dim pad">leaderboard unavailable: ${esc(e.message)} — <a href="#" onclick="loadHistory();return false">retry</a></div>`;
    }
    // confidence calibration (stated vs realised)
    try {
      const cal = await api("/api/history/calibration");
      $("#calib-meta").textContent = `n=${cal.n} · brier ${cal.brier ?? "—"} · hit ${cal.hit_rate != null ? Math.round(cal.hit_rate * 100) + "%" : "—"}`;
      const box = $("#hs-calib");
      if (!cal.n) {
        box.innerHTML = `<div class="dim pad">needs settled windows with a locked confidence — n=0 / ${cal.min_n} (honest: nothing to calibrate yet)</div>`;
      } else {
        const rows = (cal.bins || []).map(b => {
          const w = Math.min(100, (b.hit_rate ?? 0) * 100);
          const exp = Math.round(b.conf * 100);
          return `<div class="contrib" style="grid-template-columns:64px 1fr 96px">
            <span class="cname">${Math.round(b.conf * 100)}% conf</span>
            <div class="bar" title="realised hit-rate"><i class="${(b.hit_rate ?? 0) >= b.conf - 0.1 ? "pos" : "neg"}" style="width:${Math.max(2, w)}%"></i>
              <u style="position:absolute;left:${exp}%;top:0;bottom:0;width:1px;background:var(--gold)"></u></div>
            <span class="val">${b.hit_rate != null ? Math.round(b.hit_rate * 100) + "%" : "—"} <span class="dim">n${b.n}</span></span>
          </div>`;
        }).join("");
        box.innerHTML = rows +
          `<div class="hint">gold tick = the confidence the engine STATED; bar = how often that bucket actually hit (strict sign, tie=loss). ${esc(cal.note || "")} ${cal.enough ? "" : `<b class="acc-warn">preliminary — needs n≥${cal.min_n}</b>`}</div>`;
      }
    } catch { /* panel optional */ }
    const csvBtn = $("#hs-csv");
    if (csvBtn && !csvBtn.dataset.wired) {
      csvBtn.dataset.wired = "1";
      csvBtn.addEventListener("click", () => {
        let csv = "kind,key,value\n";
        (lb.leaderboard || []).forEach(r => {
          csv += `leaderboard,${r.name || r.formula},${r.accuracy}\n`;
        });
        dd.forEach(d2 => {
          csv += `day,${d2.date},grcs=${d2.grcs};pred=${d2.predictions};hits=${d2.targets_hit};sls=${d2.sls_hit}\n`;
        });
        const blob = new Blob([csv], { type: "text/csv" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "history-export.csv";
        a.click();
        URL.revokeObjectURL(a.href);
        toast("history CSV exported", "ok");
      });
    }
    if (dd.length && !S.histDay) showDay(dd[0].date);
    else if (S.histDay) showDay(S.histDay);
  } catch (e) { toast("history: " + e.message, "err"); }
}
async function showDay(date) {
  try {
    S.histDay = date;
    $("#hs-day-label").textContent = date;
    const d = await api(`/api/history/day?date=${date}`);
    // records are per-GRC: {timestamp, predictions:{tf:{...}}, outcomes:{tf:{...}}}
    const recs = d.records || d.rows || [];
    const rows = [];
    recs.forEach(rc => {
      const preds = rc.predictions || {};
      Object.keys(preds).forEach(tf => {
        const pr = preds[tf] || {};
        const oc = (rc.outcomes || {})[[tf]] || {};
        rows.push({
          tf, direction: pr.direction ?? 0,
          confidence: pr.confidence, issued: pr.issued,
          result: oc.target_hit ? "target ✓" : oc.sl_hit ? "SL ✗"
            : (pr.issued ? "open / no close" : "watch-only"),
          time: Date.parse(rc.timestamp || "") / 1000 || null,
        });
      });
    });
    $("#hs-detail").innerHTML = rows.length ? `
      <div class="day-row head"><span>TF</span><span>DIR</span><span>CONF</span><span>RESULT</span><span>TIME</span></div>` +
      rows.slice(-40).reverse().map(r => `
        <div class="day-row">
          <span>${(r.tf || "").toUpperCase()}</span>
          <span style="color:${r.direction > 0 ? "var(--green)" : r.direction < 0 ? "var(--red)" : "var(--dim)"}">${r.direction > 0 ? "▲" : r.direction < 0 ? "▼" : "●"}</span>
          <span>${r.confidence != null ? fx(r.confidence, 3) : "—"}</span>
          <span>${r.result}</span>
          <span class="dim">${r.time ? new Date(r.time * 1000).toLocaleTimeString() : ""}</span>
        </div>`).join("")
      : `<div class="dim pad">no prediction rows for this day</div>`;
  } catch (e) { toast("day: " + e.message, "err"); }
}

/* ═══════════ SETTINGS ═══════════ */
async function loadRL() {
  const box = $("#rl-panel"); if (!box) return;
  try {
    const rl = await api("/api/rl/status");
    $("#rl-meta").textContent = `threshold ${fmt(rl.threshold, 2)} · build v${rl.version}`;
    const bandit = rl.bandit || {};
    const bstats = Object.entries(bandit.stats || {}).map(([k, v]) =>
      `<div class="kv"><span>threshold ${fmt(parseFloat(k), 2)}</span><b>${v.n} settled · Σr ${fmt(v.reward, 1)}</b></div>`).join("");
    const damps = Object.entries(rl.damp || {});
    const fixes = (rl.active_fixes || []).map(a =>
      `<div class="kv"><span>${esc(a.kind)} F${a.id}</span><b class="acc-warn">${esc(a.action)} <span class="dim">reward ${fmt(a.reward, 2)} n=${a.n}</span></b></div>`).join("");
    const acts = (rl.recent_actions || []).map(a =>
      `<div class="dim">${new Date(a.at * 1000).toLocaleTimeString()} · ${esc(a.kind)} ${a.id ? "F" + a.id : ""} · ${esc(a.action || "")}${a.reward != null ? " · r " + fmt(a.reward, 2) : ""}</div>`).join("");
    const t = rl.target || {};
    box.innerHTML = `
      <div class="kv-row">
        <div class="kv"><span>issue threshold (bandit)</span><b>${fmt(rl.threshold, 2)}</b></div>
        <div class="kv"><span>committed</span><b>${bandit.committed ?? 0}/${bandit.commit_n ?? 8} windows</b></div>
        <div class="kv"><span>accuracy TARGET</span><b>${Math.round((t.target ?? 0.9) * 100)}%</b></div>
        <div class="kv"><span>measured now</span><b class="${t.current != null && t.current >= (t.target ?? 0.9) ? "acc-up" : "acc-warn"}">${t.current != null ? Math.round(t.current * 100) + "%" : "—"} <span class="dim">n=${t.n ?? 0}${t.gap_pp != null ? " · gap " + t.gap_pp + "pp" : ""}</span></b></div>
      </div>
      <div class="hint" style="${t.status && String(t.status).includes("NOT MET") ? "color:var(--gold)" : ""}">${esc(t.status || "")} — RL climbs with real settlements; the number is never rewritten.</div>
      <div class="panel-h" style="margin-top:8px">ACTIVE WEIGHT DAMPS (error-finder)</div>
      <div class="kv-row">${damps.length ? damps.map(([id, v]) =>
        `<div class="kv"><span>F${id}</span><b class="acc-warn">×${fmt(v, 2)}</b></div>`).join("")
        : `<div class="dim">none — every graded component's reward ≥0.40</div>`}</div>
      ${fixes ? `<div class="kv-row">${fixes}</div>` : ""}
      <div class="panel-h" style="margin-top:8px">WINNER BOOSTS (reward &gt;0.60 earns weight, cap ×1.50)</div>
      <div class="kv-row">${Object.entries(rl.boost || {}).length ? Object.entries(rl.boost).map(([id, v]) =>
        `<div class="kv"><span>F${id}</span><b class="acc-up">×${fmt(v, 2)}</b></div>`).join("")
        : `<div class="dim">none yet — no component has sustained reward &gt;0.60 over ≥8 settled windows</div>`}</div>
      <div class="panel-h" style="margin-top:8px">ISSUE-BANDIT SCOREBOARD</div>
      <div class="kv-row">${bstats}</div>
      <div class="panel-h" style="margin-top:8px">CONTEXTUAL BANDITS (asset|tf|vol regime — own data or global)</div>
      <div class="kv-row">${Object.entries(rl.contexts || {}).length ? Object.entries(rl.contexts).map(([ctx, b]) =>
        `<div class="kv"><span>${esc(ctx)}</span><b>thr ${fmt(b.threshold, 2)} · ${b.committed ?? 0}/8</b></div>`).join("")
        : `<div class="dim">no context has settled windows yet — every context uses the global bucket</div>`}</div>
      <div class="panel-h" style="margin-top:8px">Q-TABLE — subsystem mix per window state (decided windows only)</div>
      <div class="kv-row">${Object.entries(rl.qtable || {}).length ? Object.entries(rl.qtable).map(([state, cell]) =>
        `<div class="kv"><span>${esc(state)}</span><b>${Object.entries(cell).map(([a, v]) => `${esc(a)} Q${fmt(v.q, 2)} n=${v.n}`).join(" · ")}</b></div>`).join("")
        : `<div class="dim">no decided windows yet — every state plays the balanced (legacy) mix</div>`}</div>
      ${acts ? `<div class="hint" style="margin-top:6px">recent RL actions:<br/>${acts}</div>` : ""}`;
  } catch (e) {
    box.innerHTML = `<div class="dim pad">rl console: ${esc(e.message)}</div>`;
  }
}

/* ═══════ Binance execution (optional, gated — app/trading.py) ═══════ */
async function loadTrading() {
  try {
    const t = await api("/api/trading/status");
    const c = t.config || {};
    $("#tr-status").textContent = !c.enabled ? "OFF (default)"
      : !c.has_keys ? "enabled but NO KEYS"
      : `${c.testnet ? "TESTNET" : "MAINNET ⚠"} · ${c.armed ? "ARMED" : "not armed"}`;
    $("#tr-open").textContent = Object.keys(t.open_positions || {}).join(", ") || "none";
    $("#tr-today").textContent = String(t.orders_today ?? 0);
    if ($("#tr-key") && c.api_key) $("#tr-key").placeholder = c.api_key;
    if ($("#tr-max") && c.max_order_usdt) $("#tr-max").value = c.max_order_usdt;
    if ($("#tr-cap") && c.max_orders_day) $("#tr-cap").value = c.max_orders_day;
    if ($("#tr-testnet")) $("#tr-testnet").checked = c.testnet !== false;
    if ($("#tr-enabled")) $("#tr-enabled").checked = !!c.enabled;
    const log = $("#tr-log");
    if (log) log.innerHTML = (t.log || []).slice(-8).reverse().map(r =>
      `<div>${new Date((r.ts || 0) * 1000).toTimeString().slice(0, 8)} ${esc(r.symbol || "")} ${esc(r.tf || "")} — <b class="${r.ok ? "acc-up" : "acc-dn"}">${r.ok ? "FIRED" : "refused"}</b> ${esc(r.why || "")}</div>`).join("") || "no decisions yet";
  } catch { /* trading optional */ }
}
function bindTrading() {
  const save = $("#tr-save"), arm = $("#tr-arm"), dis = $("#tr-disarm"), clr = $("#tr-clear");
  if (!save) return;
  save.addEventListener("click", async () => {
    try {
      await api("/api/trading/config", { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: $("#tr-key").value, api_secret: $("#tr-secret").value,
          max_order_usdt: Number($("#tr-max").value || 25), max_orders_day: Number($("#tr-cap").value || 4),
          testnet: $("#tr-testnet").checked, enabled: $("#tr-enabled").checked }) });
      toast("trading settings saved (keys masked, stored locally)", "ok");
    } catch (e) { toast("trading: " + e.message, "err"); }
    loadTrading();
  });
  arm.addEventListener("click", async () => {
    await api("/api/trading/arm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ minutes: 30 }) });
    toast("ARMED for 30 min — only ISSUED signals can fire", "ok"); loadTrading();
  });
  dis.addEventListener("click", async () => {
    await api("/api/trading/arm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ minutes: 0 }) });
    toast("DISARMED — no orders can fire", "warn"); loadTrading();
  });
  clr.addEventListener("click", async () => {
    await api("/api/trading/config", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ clear_keys: true }) });
    toast("keys cleared, trading disabled", "warn"); loadTrading();
  });
}

async function loadDemoSync() {
  const badge = $("#demo-sync-badge");
  if (!badge) return;
  try {
    const r = await api("/api/demo-sync/status");
    badge.textContent = r.badge || "APP PREDICTION LOCKS (SIMULATED) — NO TESTNET KEYS LINKED";
    badge.className = r.linked ? "acc-up" : "acc-warn";
    if ($("#ds-testnet")) $("#ds-testnet").checked = r.testnet !== false;
    if ($("#ds-market") && r.market) $("#ds-market").value = r.market;
    const updated = $("#ds-updated");
    if (updated) updated.textContent = r.fetched_at
      ? `${r.cached ? "cached" : "polled"} ${new Date(r.fetched_at * 1000).toLocaleTimeString()} · read-only`
      : "not polled";
    const rec = $("#ds-reconcile");
    const ledgerSync = $("#demo-sync-ledger");
    if (!r.linked && ledgerSync) ledgerSync.textContent = r.badge || "APP PREDICTION LOCKS (SIMULATED) — NO TESTNET KEYS LINKED";
    else if (r.linked && ledgerSync) {
      const mismatches = (r.reconciliation || []).filter(x => String(x.status).includes("MISMATCH") || String(x.status).includes("no matching"));
      ledgerSync.innerHTML = `<b>${esc(r.badge || "READ-ONLY ACCOUNT LINKED")}</b> · ${r.positions?.length || 0} positions · ${r.fills?.length || 0} recent fills · ${mismatches.length} lock/position warning(s)` +
        (mismatches.length ? `<div>${mismatches.slice(0, 6).map(x => `${esc(x.asset)} ${esc(x.timeframe)}: ${esc(x.status)}`).join(" · ")}</div>` : "") +
        `<div class="hint">Heuristic asset/side comparison only; no order ID link to prediction locks.</div>`;
    }
    if (rec) {
      if (!r.linked) rec.textContent = "No external account is linked. Internal prediction locks are local measurements and are not actual fills.";
      else if (!r.ok) rec.textContent = `Account poll failed: ${r.error || "unknown error"}`;
      else {
        const rows = r.reconciliation || [];
        rec.innerHTML = `<div class="kv-row">${rows.slice(0, 16).map(x =>
          `<div class="kv"><span>${esc(x.asset)} ${esc(String(x.timeframe || "").toUpperCase())} · lock ${esc(x.lock_side || "—")} / ${esc(x.lock_status || "")}</span><b>${esc(x.status)}${x.account_entry != null ? ` · acct entry ${fmt(x.account_entry, 4)} qty ${fmt(x.account_quantity, 5)}` : ""}</b></div>`
        ).join("") || `<div class="dim">No internal prediction locks yet.</div>`}</div>
        <div class="hint">Comparison is asset/side heuristic only; no external order ID is linked to a prediction lock.</div>`;
      }
    }
    const fills = $("#ds-fills");
    if (fills && r.linked && r.ok) {
      fills.innerHTML = (r.fills || []).slice(0, 20).map(f =>
        `<div>${new Date((f.time_ms || 0)).toLocaleString()} · ${esc(f.asset)} ${esc(f.side)} ${fmt(f.quantity, 5)} @ ${fmt(f.price, 5)} · realized ${f.realized_pnl == null ? "n/a (spot endpoint)" : fmt(f.realized_pnl, 4)} · fee ${fmt(f.commission, 6)} ${esc(f.commission_asset || "")}</div>`
      ).join("") || "No recent fills returned by the exchange.";
    } else if (fills && !r.linked) fills.textContent = "No keys linked — no account polling performed.";
  } catch (e) {
    if (badge) badge.textContent = `DEMO SYNC UNAVAILABLE — ${e.message}`;
  }
}
function bindDemoSync() {
  const save = $("#ds-save"), refresh = $("#ds-refresh"), clear = $("#ds-clear");
  if (!save) return;
  save.addEventListener("click", async () => {
    try {
      await api("/api/demo-sync/config", { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: $("#ds-key").value.trim(), api_secret: $("#ds-secret").value.trim(),
          testnet: $("#ds-testnet").checked, market: $("#ds-market").value }) });
      $("#ds-key").value = ""; $("#ds-secret").value = "";
      toast("read-only sync credentials saved locally; no orders can be placed by this module", "ok");
    } catch (e) { toast("demo sync: " + e.message, "err"); }
    loadDemoSync();
  });
  refresh?.addEventListener("click", async () => {
    try {
      await api("/api/demo-sync/config", { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ testnet: $("#ds-testnet").checked, market: $("#ds-market").value }) });
      await loadDemoSync();
    } catch (e) { toast("demo sync poll: " + e.message, "err"); }
  });
  clear?.addEventListener("click", async () => {
    try {
      await api("/api/demo-sync/config", { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clear_keys: true }) });
      toast("read-only sync keys cleared", "warn"); loadDemoSync();
    } catch (e) { toast("demo sync: " + e.message, "err"); }
  });
}

async function loadSettings() {
  try {
    loadRL().catch(() => {});
    loadTrading().catch(() => {});
    loadDemoSync().catch(() => {});
    const l = await api("/api/learning");
    const lr = l.learning || {};
    $("#st-eta").textContent = fmt(lr.eta ?? lr.rates?.eta ?? 0.02, 3);
    $("#st-alert").textContent = lr.high_alert ? "ON" : "off";
    $("#st-shares").textContent = Object.entries(l.shares || {}).map(([k, v]) => `${k}:${fmt(v, 2)}`).join(" · ") || "—";
  } catch { /* optional */ }
}

function syncSystemBtn() {
  const b = $("#btn-system");
  if (!b) return;
  const m = (S.systemMode || "old").toUpperCase();
  // "SYSTEM: OLD" read as "this build is out of date" (the version watch
  // reloads on a new commit, so there was nothing to update). It is a MODEL
  // selector, so the label says which one is selected and the tooltip says
  // what tapping does instead of leaving the ambiguity to the reader.
  b.textContent = `MODEL: ${m}`;
  b.title = `prediction model in use: ${m} — tap to switch to ` +
            (m === "OLD" ? "NEW" : "OLD");
}
/* ═══════════ boot overlay + timers ═══════════ */
let bootDeadline = 0;
function renderBoot(snap) {
  const ov = $("#boot-overlay");
  const chip = $("#boot-chip");
  if (snap.phase !== "analysis") {
    ov.classList.add("gone"); chip?.classList.add("hidden"); return;
  }
  if (S.bootDismissed) {          // user chose the dashboard; never re-block
    ov.classList.add("gone");
    if (chip) { chip.classList.remove("hidden"); chip.textContent = `⏳ analysis ${$("#boot-clock")?.textContent || ""} — details`; }
    return;
  }
  ov.classList.remove("gone");
  const an = snap.analysis || {};
  bootDeadline = Date.now() + (an.remaining_s ?? 0) * 1000;
  const rem = an.remaining_s ?? 0;
  $("#boot-clock").textContent = `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
  const ready = an.readiness || {};
  if (chip && snap.phase === "analysis" && S.bootDismissed)
    chip.textContent = `⏳ analysis ${Math.floor((an.remaining_s ?? 0) / 60)}:${String(Math.floor((an.remaining_s ?? 0) % 60)).padStart(2, "0")} — details`;
  $$("#boot-checks li").forEach(li => {
    const k = li.dataset.k;
    const ok = ready[k] === true || k === "price" ||
      (k === "bgn" && ready.formulas) || (k === "aggregation" && ready.weights);
    li.classList.toggle("ok", !!ok);
  });
  $("#boot-warning").classList.toggle("hidden", !an.warning);
}
/* ═══════ THE one setInterval (advice H-2): MASTER ASYNC SCHEDULER
   (directive §1.2).  The interval fires every 250 ms but the body runs
   on ABSOLUTE deadlines (drift-compensated): `nextAt += 1000` keeps the
   1 Hz cadence anchored to wall time, so timer drift never accumulates
   (loop_lag_ms is measured, and a >4 s stall — tab sleep — resyncs
   instead of bursting).  Async work goes through schedRun(): an in-flight
   lock per key with latest-wins coalescing, so a slow /api/snapshot can
   never queue up behind itself.  Multiple timers were the "glitchy
   countdown" bug — this is the ONLY interval. ═══ */
let _tickN = 0;
const SCHED = {
  nextAt: Date.now() + 1000,
  lagMs: 0,
  locks: {},        // key -> in_flight
  pending: {},      // key -> latest queued fn (latest wins)
  runs: 0,
};
function schedRun(key, fn) {
  if (SCHED.locks[key]) { SCHED.pending[key] = fn; return; }  // latest wins
  SCHED.locks[key] = true;
  Promise.resolve().then(fn).catch(() => {}).finally(() => {
    SCHED.locks[key] = false;
    const next = SCHED.pending[key];
    if (next) { delete SCHED.pending[key]; schedRun(key, next); }
  });
}
setInterval(() => {
  const _nowMs = Date.now();
  if (_nowMs < SCHED.nextAt) return;           // not due — fast lane idle
  SCHED.lagMs = _nowMs - SCHED.nextAt;         // measured, not guessed
  SCHED.nextAt += 1000;
  if (SCHED.nextAt < _nowMs - 4000) SCHED.nextAt = _nowMs + 1000;  // sleep resync
  SCHED.runs++;
  _tickN++;
  // realised-path ring for the probability-branches chart (1 Hz, capped)
  try {
    const _pp = S.prices?.[S.symbol];
    if (_pp > 0) {
      if (!S.priceHist) S.priceHist = [];
      S.priceHist.push({ t: serverNow(), p: _pp });
      if (S.priceHist.length > 900) S.priceHist = S.priceHist.slice(-900);
    }
  } catch (e2) { /* ring optional */ }
  if (S.snap?.system_mode && S.snap.system_mode !== S.systemMode) {
    S.systemMode = S.snap.system_mode; syncSystemBtn();
  }
  // 1) boot countdown
  const ov = $("#boot-overlay");
  if (ov && !ov.classList.contains("gone") && bootDeadline) {
    const rem = Math.max(0, (bootDeadline - Date.now()) / 1000);
    const bc = $("#boot-clock");
    if (bc) bc.textContent = `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
  }
  // 2) REST safety net every other tick (~2s) — keeps header honest.
  //    Coalesced: one snapshot request in flight at a time, latest wins.
  if (_tickN % 2 === 0 && Date.now() - S.lastMsgAt >= 4000) {
    schedRun("snapshot", () => api("/api/snapshot").then(applySnapshot));
  }
  if (_tickN % 5 === 0) { pollChartLink(); pollBacktest(); }
  // issue 3: analyse REST reconciliation runs on THIS timer (every ~2 s
  // while a poll budget is open) — never a second setInterval (H-2)
  if (_tickN % 2 === 0 && S._anPollUntil && Date.now() < S._anPollUntil) {
    _analysePollStep().catch(() => {});
  }
  if (_tickN % 30 === 15) {           // version watch: auto-reload
    api("/api/health").then(h => {
      // scheduler liveness (measured, not guessed): a silent GRC clock is
      // NOT an offline engine — surface it as its own state and let the
      // server-side watchdog heal it
      if (h.clock) S.clockInfo = h.clock;
      if (h.brain?.state) S.brainLive = h.brain;
      if (!h.commit || h.commit === "?") return;
      if (!S._commit) { S._commit = h.commit; return; }
      if (h.commit !== S._commit && !S._reloaded) {
        S._reloaded = true;
        toast("new version live — refreshing…", "ok");
        setTimeout(() => location.reload(), 600);
      }
    }).catch(() => {});
  }
  // 2b) candle series refresh (~30 s): the realised line's backfill
  // must stay current or the charts drift from the tape again —
  // coalesced so a slow reload never stacks on the next due tick
  if (_tickN % 30 === 0) schedRun("candles", loadCandles);
  // HOLD POLICY cells: countdowns tick every second; the 16-cell state
  // refetches every ~15 s (and immediately on analyse/lock events)
  if (_tickN % 15 === 7) { loadCells().catch(() => {}); }
  updateCellsTick();
  // connection honesty on the chip while the socket is silent.
  // A dead websocket with a healthy REST poll is NORMAL behind the
  // Codespaces/mobile proxy — say "polling", don't accuse the engine.
  const _fc = $("#feed-chip");
  const _wsLive = ws && ws.readyState === 1;
  const _httpFresh = S.lastHttpAt && Date.now() - S.lastHttpAt < 10000;
  const _msgFresh = S.lastMsgAt && Date.now() - S.lastMsgAt < 15000;
  if (_fc) {
    if (!_wsLive && !_msgFresh && !_httpFresh) {
      if (!_fc.dataset.stale) {
        _fc.dataset.stale = "1";
        _fc.textContent = "⚠ connection stale — retrying";
        _fc.className = "feed-chip stale";
      }
    } else if (_fc.dataset.stale || !_fc.dataset.mode) {
      delete _fc.dataset.stale;
      _fc.dataset.mode = _wsLive ? "live" : "polling";
      _fc.textContent = _wsLive ? "live" : "polling · ws off";
      _fc.className = "feed-chip" + (_wsLive ? "" : " poll");
    }
  }
  // 3) §10.4 silent-socket guard (inside THE one interval, H-2): a socket
  // with no message for 15s is closed — onclose reconnects immediately
  // instead of waiting the browser's 30-minute timeout (error J-7)
  if (ws && ws.readyState === 1 && S.wsLast &&
      Date.now() - S.wsLast > 15000) {
    S.wsLast = 0;
    try { ws.close(); } catch {}
  }
  // 3) quadrant countdown rings (server-anchored)
  _quadTick();
  // §1.4 header countdowns (server snapshot clock)
  try { renderSyncChips(); } catch (e3) { /* chips optional */ }
}, 1000);

function renderSyncChips() {
  // §1.4 the 5 named countdowns from the shared snapshot clock — one
  // line in the header, full detail in the tooltip
  const el = $("#sync-chips");
  const sync = S.snap?.sync;
  if (!el || !sync) return;
  const pick = (k) => (sync[k] && sync[k].remaining) ? sync[k].remaining : "—";
  const wc = sync.WINDOW_CLOSE_SETTLE;
  el.textContent = `⏱ tick ${pick("NEXT_TICK_PASS")} · close ${pick("WINDOW_CLOSE_SETTLE")} · brain ${pick("BRAIN_METACOG_REWEIGHT")} · comm ${pick("AGENT_COMMITTEE_SYNTHESIS")} · news ${pick("NEWS_MACRO_POLL")}`;
  el.title = `shared snapshot clock (${sync.anchor || "utc"}) — snapshot_ts ${sync.snapshot_ts ? new Date(sync.snapshot_ts * 1000).toISOString().slice(11, 19) : "?"}`
    + (wc ? ` · ${wc.label || "window close"} in ${wc.remaining}` : "")
    + " · NEXT_TICK_PASS · BRAIN_METACOG_REWEIGHT · AGENT_COMMITTEE_SYNTHESIS · NEWS_MACRO_POLL";
}

function _quadTick() {
  const now = Date.now() / 1000;
  const a = S.snap?.assets?.[S.symbol] || {};
  TF.forEach(tf => {
    const cyc = a.tfs?.[tf]; if (!cyc) return;
    const q = $("#q-" + tf); if (!q) return;
    const total = Math.max(1, (cyc.cycle_end || 0) - (cyc.cycle_start || 0));
    // server-anchored remaining time (advice H-2): re-anchored each snapshot
    const serverNow = (Date.now() + (S.serverOffset || 0)) / 1000;
    const rem = clamp((cyc.cycle_end || 0) - serverNow, 0, total);
    const fg = $(".ring-fg", q);
    if (fg) fg.style.strokeDashoffset = (97.4 * (1 - rem / total)).toFixed(1);
    const t = $(".q-time", q);
    if (t) t.textContent = `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
    if (rem <= 10 && rem > 0 && S.lastSec[tf] !== Math.ceil(rem)) {
      S.lastSec[tf] = Math.ceil(rem); haptic("last10");
    }
    if (cyc.cycle_start && S.grcFlash[tf] !== cyc.cycle_start) {
      if (S.grcFlash[tf]) { q.style.boxShadow = "0 0 0 2px #00d4f066"; setTimeout(() => q.style.boxShadow = "", 550); haptic("grc"); }
      S.grcFlash[tf] = cyc.cycle_start;
    }
  });
}


function haptic(kind) {
  try {
    const nav = S.snap?.settings?.haptics;
    if (nav === false) return;
    if (kind === "last10" && navigator.vibrate) navigator.vibrate(8);
    if (kind === "grc" && navigator.vibrate) navigator.vibrate([6, 40, 6]);
  } catch { /* no vibration API */ }
}

/* ═══════════ alert bar (SL/SAFE one-shots) ═══════════ */
function renderAlert(snap) {
  const cyc = snap.assets?.[S.symbol]?.tfs?.[S.focusTf];
  const bar = $("#alert-bar");
  if (!cyc) { bar.classList.add("hidden"); return; }
  if (cyc.close_alert) {
    bar.className = "sl"; $("#alert-icon").textContent = "▼";
    $("#alert-text").textContent = "CLOSE — TARGET/SL HIT · FEEDBACK APPLIED";
    setTimeout(() => bar.classList.add("hidden"), 9000);
  } else if (cyc.safe_alert) {
    bar.className = "safe"; $("#alert-icon").textContent = "▲";
    $("#alert-text").textContent = "SAFE — SL TIGHTENED 15%";
    setTimeout(() => bar.classList.add("hidden"), 9000);
  }
}

async function repairChartLink(manual) {
  Own.key = ""; Own.rangeKey = ""; Own.lineKey = ""; Own.markerKey = "";
  try { Bn.main = null; Bn.quads = {}; } catch {}
  await loadCandles().catch(() => {});
  try { bnEnsure(); } catch {}
  try { drawMainChart(); } catch {}
  if (manual) toast("chart link repaired — candles resynced to the live tape", "ok");
}
async function pollChartLink() {
  try {
    const d = await api("/api/chart-link");
    const box = $("#link-panel"); if (!box) return;
    const rows = Object.values(d.assets || {}).map(r => `
      <tr><td class="mono">${r.symbol}</td>
      <td class="mono">${fmt(r.live_price, 2)}</td>
      <td class="mono">${fmt(r.last_bar_close, 2)}</td>
      <td class="mono" style="color:${r.ok ? "var(--green)" : "var(--red)"}">${r.delta_bps == null ? "—" : r.delta_bps.toFixed(1) + " bps"}</td>
      <td class="mono">${r.bar_age_s == null ? "—" : Math.round(r.bar_age_s) + "s"}</td>
      <td class="mono" style="color:${r.foreign_bars ? "var(--red)" : "var(--green)"}">${r.foreign_bars ?? "?"}</td>
      <td class="mono" style="color:${r.history_jump_bps == null ? "var(--dim)"
        : r.history_jump_bps > (r.jump_limit_bps ?? 300) ? "var(--red)" : "var(--green)"}">${r.history_jump_bps == null ? "—" : r.history_jump_bps.toFixed(1) + " bps"}</td>
      <td>${r.ok ? "✓ linked" : "✗ UNTRUSTED"}</td></tr>`).join("");
    const inc = (d.watchdog?.incidents || []).slice(-3).reverse().map(i =>
      `<div class="hint">⚠ ${new Date(i.ts * 1000).toLocaleTimeString()} ${i.symbol} untrusted: ${esc((i.integrity_reasons || []).join(", "))}; quote delta ${i.delta_bps?.toFixed?.(1) ?? "—"} bps, candle jump ${i.history_jump_bps?.toFixed?.(1) ?? "—"} bps (${esc(i.mode)}/${esc(i.source)})</div>`).join("");
    box.innerHTML = `<table class="tbl"><thead><tr><th>ASSET</th><th>LIVE</th><th>LAST CANDLE</th><th>Δ</th><th>BAR AGE</th><th>FOREIGN</th><th>MAX JUMP</th><th>LINK</th></tr></thead><tbody>${rows}</tbody></table>${inc}`;
    const meta = $("#link-meta");
    if (meta) meta.textContent = `server-measured · ${d.watchdog?.checks ?? 0} checks · ${d.watchdog?.desync_checks ?? 0} desync checks`;
    const f = (d.assets || {})[S.symbol];
    if (f && !f.ok && Date.now() - (S._linkRepairAt || 0) > 30000) {
      S._linkRepairAt = Date.now();       // auto-repair, ≤ 1 / 30 s
      repairChartLink(false);
    }
  } catch { /* endpoint warming up */ }
}

async function pollBacktest() {
  const now = Date.now();                    // 5-min self-throttle
  if (S._btAt && now - S._btAt < 290000) return;
  S._btAt = now;
  try {
    const d = await api("/api/backtest");
    const box = $("#bt-panel"); if (!box) return;
    if (!d.ok) { box.innerHTML = `<div class="dim pad">${esc(d.reason || "no real backtest yet")}</div>`; return; }
    const row = (n, m) => m && m.n ? `<tr><td class="mono">${n}</td><td class="mono">${m.n}</td><td class="mono">${m.win_pct}%</td><td class="mono">${m.breakeven_pct}%</td><td class="mono" style="color:${m.ev_r >= 0 ? "var(--green)" : "var(--red)"}">${m.ev_r >= 0 ? "+" : ""}${m.ev_r}R</td><td class="mono">${m.profit_factor}</td><td class="mono">${m.max_dd_pct}%</td></tr>` : "";
    box.innerHTML = `<table class="tbl"><thead><tr><th>ASSET</th><th>n</th><th>WIN%</th><th>BREAKEVEN</th><th>EV</th><th>PF</th><th>MAXDD</th></tr></thead><tbody>${Object.entries(d.per_asset || {}).map(([a, m]) => row(a, m)).join("")}${row("ALL", d.combined)}</tbody></table>`;
    const meta = $("#bt-meta"); if (meta) meta.textContent = d.source || "";
  } catch { /* warming up */ }
}
pollBacktest();   // repeats via the tick loop below (one clock, H-2)

/* ═══════════ snapshot / tick plumbing ═══════════ */
// s.serverOffset re-anchors on EVERY message (advice H-2): the countdown
// never drifts with the client's wall clock.
/* alert the moment a call actually ISSUES (crosses the floor) —
   sub-floor leans stay silent by design */
function checkIssuedAlerts(snap) {
  if (!S._issuedSeen) S._issuedSeen = {};
  for (const [sym, a] of Object.entries(snap.assets || {})) {
    for (const [tf, c] of Object.entries(a.tfs || {})) {
      const k = `${sym}|${tf}`;
      const was = S._issuedSeen[k];
      // alert only on a real ISSUE verdict — `issued` is contract-true
      const isNow = verdictOf(c, null) === "ISSUED";
      S._issuedSeen[k] = isNow;
      if (isNow && was === false) {
        toast(`⚡ ISSUED ${sym} ${tf.toUpperCase()} ${c.direction > 0 ? "▲ BULL" : "▼ BEAR"} · entry ${px(c.entry)} · TP ${px(c.tp)} · SL ${px(c.sl)}`, "ok");
      }
    }
  }
}

function applySnapshot(snap) {
  if (!snap || snap.type !== "snapshot") return;
  S.snap = snap;
  if (snap.ts) S.serverOff = snap.ts - Date.now() / 1000;
  // snapshot = price truth only when the tick stream is dead (>5 s)
  const _tf = S._lastTickAt && (Date.now() - S._lastTickAt < 5000);
  for (const [sym, av] of Object.entries(snap.assets || {}))
    if (av.price > 0 && (!_tf || !((S.prices || {})[sym] > 0)))
      (S.prices ||= {})[sym] = av.price;
  try { checkIssuedAlerts(snap); } catch (e) { /* alerts optional */ }
  S.serverOffset = (snap.ts ? snap.ts * 1000 : Date.now()) - Date.now();
  renderBoot(snap);   // overlay FIRST
  renderAlert(snap);
  try { renderQuads(); renderWhy(); renderRail(); } catch (e) { console.error("render", e); }
  S.lastHttpAt = Date.now();   // REST proof of life (banner + feed chip)
  try { renderBranches(); } catch (e) { console.error("branches", e); }
  try { drawMainChart(); } catch (e) { console.error("main chart", e); }
  try { const p2 = S.prices?.[S.symbol];
    if (p2 > 0) TF.forEach(tf => S.charts[tf]?.tick(p2)); } catch (e) { /* quads optional */ }
  try { renderRiskCalc(); } catch (e) { /* sizing optional */ }
  try { applyChartSrc(); } catch (e) { /* chart toggle optional */ }
  if (S.view === "dashboard") {
    try { renderMicro(); } catch (e) { console.error("micro", e); }
  }
  try { tickFormulaValues(); } catch (e) { /* ticker optional */ }
  // live formula charts keep redrawing from the ring while the tab is open
  if (S.view === "formulas") {
    try { drawSparks(S.fvBackfill || {}); } catch (e) { /* sparks optional */ }
  }
  // honesty: tape chip AFTER content render (feed stats carry source/label)
  const fchip = $("#feed-chip");
  if (fchip && snap.feed) {
    const f = snap.feed;
    fchip.textContent = f.simulated ? "⚠ SIMULATED TAPE" : f.label;
    fchip.className = "feed-chip " + (f.simulated ? "sim" : f.stale ? "stale" : "live");
    // "simulated" alone reads as a setting. The chip says WHY the real feed is
    // not in use, because the honest answer is usually "this host cannot reach
    // the exchange" and the user deserves to see the attempt and its error.
    fchip.title = `source ${f.source} · last delivery ${f.last_delivery_age_s ?? "?"}s ago · ${f.updates} updates · ${f.switches} switches`
      + (f.real_reason ? `\n${f.real_reason}` : "");
  }
  // §10.3 status chips — every subsystem's honesty state, one glance
  const sc = $("#status-chips");
  if (sc && snap.chips) {
    const c = snap.chips;
    const chip = (label, cls, tip) =>
      `<span class="chip ${cls}" title="${esc(tip || "")}">${label}</span>`;
    sc.innerHTML =
      chip((c.feed === "binance" || c.feed === "live") ? "● LIVE FEED" : "⚠ SIM FEED",
           (c.feed === "binance" || c.feed === "live") ? "chip-ok" : "chip-warn",
           "tape source from /api/feeds/diagnose") +
      chip("local " + (c.local_model === "on" ? "ON" : "off"),
           c.local_model === "on" ? "chip-ok" : "",
           "local model availability") +
      chip("agents " + c.agents, c.keyless ? "" : "chip-ok",
           "cloud/ local/ drosophila mode — keyless runs brain+formulas") +
      chip("news " + c.news, c.news === "live" ? "chip-ok" : "chip-warn",
           "news source state") +
      (c.degrade > 0 ? chip("L" + c.degrade + " DEGRADED", "chip-warn",
                            "see /api/degrade") : "") +
      (c.deadline_misses ? chip(c.deadline_misses + " LATE AGENT CALLS",
                                "chip-warn",
                                "late answers discarded, never applied (I-3)")
                         : "");
    const cl = S.clockInfo;
    if (cl) {
      const bad = cl.alive === false ||
        (cl.last_fired_age_s != null && cl.last_fired_age_s > 600);
      if (bad)
        sc.innerHTML += chip("GRC SCHEDULER HEALING", "chip-warn",
          "the master clock restarts itself; the board keeps serving while it heals");
      else if (cl.restarts > 0)
        sc.innerHTML += chip("GRC SELF-HEALED ×" + cl.restarts, "chip-warn",
          "the master clock was dead or silent and was restarted automatically");
    }
  }
  refreshActiveView();
}
function onTick(msg) {
  S._lastTickAt = Date.now();
  S.prices = msg.prices || S.prices;
  const p = msg.prices?.[S.symbol];
  if (p != null) TF.forEach(tf => S.charts[tf]?.tick(p));
  drawMainChart();                  // main chart always ends at live price
  renderQuads({ heavy: false });   // 1 Hz: text only, no table rebuilds
}
function handleMessage(msg) {
  S.lastMsgAt = Date.now();
  // §1.1 atomic discard — an out-of-order snapshot or a superseded
  // analysis never touches the store (composite asset|timeframe keys,
  // monotonic state_version, latest-wins request_id)
  if (msg && msg.state_version != null) {
    if (S.lastStateVersion && msg.state_version < S.lastStateVersion) return;
    S.lastStateVersion = msg.state_version;
  }
  if (msg && (msg.type === "analyse_started" || msg.type === "analyse_stage"
              || msg.type === "analyse_done")) {
    const _cell = `${msg.asset || msg.symbol || ""}|${msg.timeframe || msg.tf || ""}`;
    if (msg.request_id) {
      S.analyseReqIds = S.analyseReqIds || {};
      const prev = S.analyseReqIds[_cell];
      if (prev && prev !== msg.request_id
          && Number(prev.split(".")[0]) > Number(msg.request_id.split(".")[0])) {
        return;                 // older request — drop the whole event
      }
      S.analyseReqIds[_cell] = msg.request_id;
    }
  }
  if (msg.type === "NEWS") onNewsTicker(msg);
  else if (msg.type === "news_emergency" || msg.type === "news_emergency_ack") {
    toast(`⚡ NEWS_EMERGENCY ${msg.asset || ""}: ${msg.reason || ""}`, "err");
    if (S.view === "news") loadNews();
  }
  else if (msg.type === "snapshot") applySnapshot(msg);
  else if (msg.type === "tick") onTick(msg);
  else if (msg.type === "phase") renderBoot({ phase: msg.phase, analysis: msg });
  else if (msg.type === "sl_event") {
    // the toast must always say WHICH trade: server messages now carry
    // it; the fallback composes it from the event fields
    const who = [msg.symbol, msg.tf && String(msg.tf).toUpperCase()]
      .filter(Boolean).join(" ");
    toast(msg.message || `${who} — ${msg.kind}`, msg.kind === "close" ? "err" : "warn");
  }
  else if (msg.type === "prediction_settled") onPredictionSettled(msg);
  else if (msg.type === "grc_progress") {
    // one frame per finished asset — keeps the sync chip alive during GRC
    S.grcProgress = msg;
    renderRail();
  }
  else if (msg.type === "analyse_started") onAnalyseStarted(msg);
  else if (msg.type === "analyse_stage") onAnalyseStage(msg);
  else if (msg.type === "analyse_done") onAnalyseDone(msg);
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  loadCandles().catch(() => {});
  try { drawMainChart(); } catch {}
});
window.addEventListener("focus", () => {
  loadCandles().catch(() => {});
  try { drawMainChart(); } catch {}
});
let ws;                       // module scope: master tick guards silence
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  try { ws = new WebSocket(`${proto}://${location.host}/ws`); } catch { return; }
  ws.onopen = () => {
    S.wsRetry = 0;
    S.wsLast = Date.now();
    try { ws.send(JSON.stringify({ type: "HELLO", view: S.view })); } catch {}
    // issue 3: if an analysis ran while the socket was down, the
    // analyse_done event is gone forever — reconcile from REST so the
    // panel never stays stuck on "calculating…"
    resyncAnalyse();
    // §10.4 silence is checked by the SINGLE master tick (H-2): a socket
    // silent for 15s is closed there — never wait the browser's 30-minute
    // timeout (error J-7)
  };
  ws.onclose = () => {
    S.wsRetry++;
    setTimeout(connectWS, Math.min(8000, 1000 * S.wsRetry));
  };
  // ONE handler. This used to be assigned twice, so the liveness stamp was
  // overwritten by the dispatch handler: S.wsLast only ever held the open
  // time, the master tick then judged the socket silent after 15 s and
  // closed it — a reconnect loop every 15 s that the user saw as "the
  // engine/brain keeps disconnecting".
  ws.onmessage = ev => {
    S.wsLast = Date.now();
    try { handleMessage(JSON.parse(ev.data)); } catch { /* partial */ }
  };
}

/* ═══════════ events ═══════════ */
function bindGlobal() {
  $("#nw-refresh")?.addEventListener("click", loadNews);
  $("#nw-drill")?.addEventListener("click", drillEmergency);
  $("#hs-reload")?.addEventListener("click", loadHistory);
  $("#mind-refresh")?.addEventListener("click", () => loadMind(true));
  $("#mind-narrate")?.addEventListener("click", async (e) => {
    const b = e.currentTarget; b.disabled = true; b.textContent = "narrating…";
    const box = $("#mind-narration");
    try {
      const n = await api(`/api/think?symbol=${S.symbol}&tf=${S.focusTf}&narrate=1`);
      const r = n.narration || {};
      if (box) {
        box.classList.remove("hidden");
        box.innerHTML = `<div class="hint">narration source: <b>${esc(r.source || "?")}</b>`
          + (r.why ? ` — ${esc(r.why)}` : "")
          + (r.guard && r.guard.ok === false ? ` · <span class="acc-dn">model text REJECTED: ${esc((r.guard.stray_numbers || []).join(", "))}</span>` : "")
          + `</div><div style="white-space:pre-wrap;font-size:11.5px">${esc(r.text || "—")}</div>`;
      }
    } catch (err) { toast("narrate: " + err.message, "err"); }
    b.disabled = false; b.textContent = "narrate";
  });
  $("#ag-reload")?.addEventListener("click", loadAgents);
  $("#ag-save")?.addEventListener("click", async () => {
    const body = {
      gemini_key: $("#key-gemini").value.trim(), backup1_key: $("#key-backup1").value.trim(),
      backup2_key: $("#key-backup2").value.trim(), backup3_key: $("#key-backup3").value.trim(),
      newsapi_key: $("#key-newsapi").value.trim(), neuprint_key: $("#key-neuprint").value.trim(),
    };
    try {
      const r = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const applied = (r?.applied || []).filter(k => k.endsWith("_key"));
      toast(applied.length ? `keys saved (${applied.join(", ")}) — press TEST on the slot`
                           : "no new key entered — stored keys are unchanged",
            applied.length ? "ok" : ""); loadAgents();
      // §10.5 (D-5): data-affecting settings force a refetch — the UI
      // never keeps showing pre-change data
      const dataKeys = ["feed_mode", "symbol", "tf", "focus_tf", "news_enabled"];
      if (Object.keys(body).some(k => dataKeys.includes(k))) {
        try { refreshActiveView(true); toast("data setting applied — view refetched", "ok"); }
        catch (e) { console.warn("refetch:", e); }
      }
    } catch (e) { toast(e.message, "err"); }
  });
}

/* ═══════════ resilience: surface errors + detect engine offline ═══════════ */
let _errShown = 0, _fetchFails = 0;      // _fetchFails = unreachable only
let _httpFails = 0;                      // one panel erroring != offline
let _busyStreak = 0;                     // timeouts = busy, never "offline"
let _bannerSince = 0;                    // when the banner first appeared
function _netBanner(show, detail) {
  let b = $("#net-banner");
  if (!show) {
    // Don't retract instantly. A busy engine answers SOME requests and times
    // out on others, so removing on the first success then re-adding made the
    // bar strobe (the flicker in the screenshot). Keep it at least 6 s, and
    // reset the counters so the next real outage still needs 3 misses.
    if (b && Date.now() - _bannerSince < 6000) return;
    if (b) b.remove();
    _fetchFails = 0; _httpFails = 0; _busyStreak = 0;
    return;
  }
  if (!b) {
    b = document.createElement("div");
    b.id = "net-banner";
    b.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:300;background:#2a0d14;" +
      "color:#f0385a;border-bottom:1px solid #f0385a;padding:8px 16px;font:600 12px monospace;" +
      "letter-spacing:1px;text-align:center";
    document.body.appendChild(b);
    _bannerSince = Date.now();
    // text and the dismiss control are separate nodes so a message update can
    // never wipe the ✕
    const msg = document.createElement("span");
    msg.className = "net-msg";
    const x = document.createElement("span");
    x.textContent = " ✕";
    x.style.cssText = "float:right;cursor:pointer;padding:0 4px";
    x.title = "dismiss (the panel keeps retrying)";
    x.addEventListener("click", () => { const el = $("#net-banner"); if (el) el.remove(); });
    b.append(msg, x);
  }
  const msgEl = b.querySelector(".net-msg");
  if (msgEl) msgEl.textContent = detail || "ENGINE OFFLINE — retrying every 2s (start it: bash run.sh)";
}
let _busyProbeAt = 0;
async function _confirmBusy() {
  const now = Date.now();
  if (now - _busyProbeAt < 3000) return;   // one probe per burst
  _busyProbeAt = now;
  try {
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), 4000);
    const r = await _origFetch("/api/health", { signal: ctl.signal });
    clearTimeout(to);
    if (r.ok) {
      // loop is answering: this is a slow panel, not a stuck engine
      const h = await r.json().catch(() => null);
      if (h && h.loop_lag_s != null && h.loop_lag_s > 2) {
        _netBanner(true, `ENGINE BUSY — loop lag ${h.loop_lag_s.toFixed(1)}s ` +
                         "(measured); panels refresh when it clears (do not restart)");
      } else {
        _busyStreak = 0;
        _netBanner(false);            // no banner theatre: nothing is wrong
      }
    }
  } catch (e) { /* health blocked too -> the next poll decides */ }
}
let _lastOkAt = Date.now();
const _origFetch = window.fetch;
window.fetch = async (...args) => {
  const isApi = String(args[0]).includes("/api/");
  try {
    const r = await _origFetch(...args);
    if (r.ok) { _fetchFails = 0; _httpFails = 0; _busyStreak = 0; _netBanner(false); }
    else if (String(args[0]).includes("/api/")) {
      _fetchFails = 0;                   // the engine answered
      if (++_httpFails >= 3)
        _netBanner(true, `ENGINE ERROR ${r.status} — a panel failed; the engine is answering`);
    }
    return r;
  } catch (e) {
    _httpFails = 0;
    // `AbortError` = we gave up waiting; the socket was never refused. That is
    // a busy engine (warm replay / GRC / model load), so say so and do NOT
    // count it toward "unreachable" — otherwise a long analysis reads as a
    // dead server and the banner tells the user to start what is running.
    const busy = !!(e && e.name === "AbortError");
    if (String(args[0]).includes("/api/")) {
      if (busy) {
        // ONE slow endpoint is not a busy engine. The banner used to fire on
        // any two aborted calls, so a single expensive panel (or a snapshot
        // rebuild) announced "ENGINE BUSY" over a perfectly responsive app.
        // Confirm with the cheapest request there is: if /api/health answers,
        // the event loop is alive and the honest report is "a panel is slow",
        // not "the engine is stuck".
        if (++_busyStreak >= 2) _confirmBusy();
      } else if (++_fetchFails >= 3) {
        _netBanner(true, "ENGINE UNREACHABLE — retrying (start it: bash run.sh)");
      }
    }
    throw e;
  }
};
// (no extra timer: the 2 s REST poll already clears the banner on success —
// the app keeps exactly ONE setInterval, per the H-2 contract)
window.addEventListener("error", e => {
  if (_errShown++ < 3) toast("JS: " + (e.message || "error"), "err");
});
window.addEventListener("unhandledrejection", e => {
  if (_errShown++ < 3) toast("async: " + (e.reason?.message || e.reason || "rejected"), "err");
});

/* ═══════════ boot ═══════════ */
(function boot() {
  try { S.tfFocus = localStorage.getItem("tfFocus") === "1"; } catch (e) { S.tfFocus = false; }
  // remember the user's asset/timeframe across reloads (the old build always
  // came back on BTC·TF1 — read as "switching did nothing")
  try {
    const sym = localStorage.getItem("symbol");
    if (sym && ASSETS.includes(sym)) S.symbol = sym;
    const tf = localStorage.getItem("focusTf");
    if (tf && TF.includes(tf)) S.focusTf = tf;
  } catch (e) { /* private mode */ }
  buildRail(); buildTabs(); buildToolbar(); buildQuads(); bindGlobal(); bindAnalyse(); bindAutoChip(); loadCells(); bindChartSrc(); bindTrading(); bindDemoSync(); bindRiskCalc(); syncAnalyseButton(); bindAllTf(); renderDashboardMode();
  connectWS();
  api("/api/snapshot").then(applySnapshot).catch(() => {});
  // REST safety net lives in the single master setInterval below (H-2)
})();

// [binance-header-btn] discoverable exchange entry point in the header.
(() => {
  const b = $("#btn-binance"); if (!b) return;
  // The MARKETS tape and Binance EXECUTION are two different things, and the
  // old button only ever opened the execution settings — while the tape chip
  // said "SIMULATED" with no way to do anything about it. Clicking now tries
  // the real market feed first (measured, and it reports the error verbatim),
  // and offers execution settings after.
  b.addEventListener("click", async () => {
    const was = b.textContent;
    b.disabled = true; b.textContent = "⌛ CONNECTING REAL FEED…";
    try {
      const r = await api("/api/feeds/connect", { method: "POST" });
      if (r.ok && r.mode === "binance") {
        toast(`real tape adopted — ${r.detail || r.source || "exchange"}`, "ok");
        refreshActiveView(true);
      } else {
        toast("real feed unavailable — " + (r.error || r.detail || "unknown") +
              " · the simulator keeps the tape", "warn");
      }
      // execution settings stay reachable regardless
      document.querySelector('.tab[data-t="settings"]')?.click();
      setTimeout(() => $("#trade-enable")?.scrollIntoView({behavior:"smooth", block:"center"}), 250);
    } catch (e) {
      toast("connect real feed: " + e.message, "err");
    } finally {
      b.disabled = false; b.textContent = was; paint();
    }
  });
  var paint = async () => {
    try { const t = await (await fetch("/api/trading/status")).json();
      const c = t.config || {};
      b.textContent = !c.enabled ? "⬢ CONNECT BINANCE"
        : !c.has_keys ? "⬢ BINANCE — KEYS NEEDED"
        : c.armed ? "⬢ BINANCE — LIVE (ARMED)" : "⬢ BINANCE — CONNECTED";
      b.style.borderColor = c.armed ? "var(--up)" : c.has_keys ? "var(--gold,#c9a227)" : "";
      b.style.color = c.armed ? "var(--up)" : "";
    } catch (e) { /* header stays as connect */ }
  };
  paint(); window.__paintBinance = paint;
})();


// [brain-header-btn] one-click brain reconnect in the header.
(() => {
  const b = $("#btn-brain"); if (!b) return;
  b.addEventListener("click", async () => {
    b.textContent = "⟳ …";
    try {
      const r = await api("/api/brain/reconnect", { method: "POST" });
      const fw = r.flywire || r;
      toast(fw.state === "ready" ? "brain reconnected — vote live"
            : "brain reconnect started — watch the chip (auto-retries too)",
            fw.state === "ready" ? "ok" : "err");
    } catch (e) { toast("brain reconnect failed — " + e.message, "err"); }
    setTimeout(() => { b.textContent = "⟳ BRAIN"; }, 1500);
  });
})();
