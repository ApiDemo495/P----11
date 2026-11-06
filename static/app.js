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
const EMO = ["Fear","Loss aversion","Hedge demand","Panic","Startle reflex",
  "Session greed","Smart-money greed","Volatility stress","Rollover panic",
  "Smart-money absorption","Herding","Range fatigue","Reversal fear",
  "Sentiment","Intent to act","Announcement impulse","Crowd pull",
  "Trend-chase fatigue","Smart-money divergence","News FOMO",
  "Trend exhaustion","Net emotion"];
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
  21:"CCI master aggregate: 36 weighted components",
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
const px = v => v >= 1000 ? Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 })
  : v >= 10 ? Number(v).toFixed(2) : Number(v).toFixed(4);
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg;
  $("#toast-wrap").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

/* ── state ──────────────────────────────────────────────────── */
const S = {
  symbol: "BTC", focusTf: "tf1", tfFocus: false, view: "dashboard",
  snap: null, prices: {}, lastMsgAt: 0, wsRetry: 0,
  cycles: {}, grcFlash: {}, lastSec: {},
  charts: {}, spark: {},
  newsSel: "BTC", histDay: null,
};

/* ═══════════ LineChart — own chart engine (v2, replaces ChartX) ═════
   One engine for the OWN main chart + sub-timeframe quads + sparklines. Design rules
   (user: "better than TradingView"): the PRICE IS ALWAYS VISIBLE —
   right-hand price axis with ticks, a last-price chip pinned to the
   axis, crosshair readout of time+price, dashed TP/SL/entry levels with
   labels, and the empirical fan cone into the future half.     */
class LineChart {
  constructor(el, opts = {}) {
    this.el = el; this.spark = !!opts.spark;
    this.cv = document.createElement("canvas");
    this.cv.style.cssText = "display:block;width:100%;height:100%";
    el.appendChild(this.cv);
    this.cross = null;
    const upd = e => { const r = this.cv.getBoundingClientRect();
      const t = (e.touches && e.touches[0]) || e;
      this.cross = { x: t.clientX - r.left, y: t.clientY - r.top }; this.draw(); };
    this.cv.addEventListener("mousemove", upd);
    this.cv.addEventListener("touchmove", upd, { passive: true });
    this.cv.addEventListener("mouseleave", () => { this.cross = null; this.draw(); });
  }
  set(data) { this.d = data; this.draw(); }
  setData(bars) { // OHLC bars -> close line (quad/spark feed)
    this.base = { pts: bars.map(b => ({ t: b.time > 1e12 ? b.time / 1000 : b.time, p: b.close })).filter(p => p.p > 0) };
    if (this.base.pts.length) { this.base.t0 = this.base.pts[0].t; this.base.t1 = this.base.pts[this.base.pts.length - 1].t; }
    this._sync(); }
  setLines(ls) { this.lines = ls || []; this._sync(); }
  setMarkers() { /* markers dropped in v2: levels+labels carry the info */ }
  tick(p) { if (!this.base || !this.base.pts.length) return;
    const now = Date.now() / 1000, last = this.base.pts[this.base.pts.length - 1];
    if (now - last.t < 1) last.p = p; else this.base.pts.push({ t: now, p });
    if (this.base.pts.length > 900) this.base.pts.splice(0, this.base.pts.length - 900);
    this.base.t1 = now; this._sync(); }
  _sync() { if (!this.base) return;
    this.d = { ...this.base, levels: (this.lines || []).map(L => ({ p: L.price, tag: L.title || "", color: L.color, dash: L.dash })) };
    this.draw(); }
  draw() {
    const d = this.d; if (!d || !d.pts || d.pts.length < 2) return;
    const cv = this.cv, W = this.el.clientWidth || 300, H = this.el.clientHeight || 120;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const root = getComputedStyle(document.documentElement);
    const INK = root.getPropertyValue("--ink").trim() || "#e8e8f2",
          DIM = root.getPropertyValue("--dim").trim() || "#8a8aa0",
          UP = "#26a69a", DN = "#ef5350", GOLD = "#c9a227";
    g.clearRect(0, 0, W, H);
    const padR = this.spark ? 4 : 58, padL = 6, padT = 8, padB = this.spark ? 4 : 16;
    const pw = W - padL - padR, ph = H - padT - padB;
    let lo = Infinity, hi = -Infinity;
    for (const p of d.pts) { lo = Math.min(lo, p.p); hi = Math.max(hi, p.p); }
    for (const L of (d.levels || [])) { lo = Math.min(lo, L.p); hi = Math.max(hi, L.p); }
    if (d.fan) for (const f of d.fan) { lo = Math.min(lo, f[2]); hi = Math.max(hi, f[1]); }
    if (!(hi > lo)) { hi = lo * 1.001; lo = lo * 0.999; }
    const pad = (hi - lo) * 0.08; hi += pad; lo -= pad;
    const t0 = d.t0, t1 = d.t1, X = t => padL + (t - t0) / Math.max(1, t1 - t0) * pw,
          Y = p => padT + (hi - p) / (hi - lo) * ph;
    // grid + price axis (the "no price visible" fix: always drawn)
    g.font = "9px ui-monospace,monospace"; g.textAlign = "left";
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const p = lo + (hi - lo) * i / ticks, y = Y(p);
      g.strokeStyle = "rgba(140,140,170,.14)"; g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + pw, y); g.stroke();
      if (!this.spark) { g.fillStyle = DIM; g.fillText(px(p), padL + pw + 4, y + 3); }
    }
    if (!this.spark) { // time axis
      g.fillStyle = DIM; g.textAlign = "center";
      for (let i = 0; i <= 4; i++) { const t = t0 + (t1 - t0) * i / 4, dt = new Date(t * 1000);
        g.fillText(`${String(dt.getHours()).padStart(2, "0")}:${String(dt.getMinutes()).padStart(2, "0")}`, X(t), H - 4); }
      g.textAlign = "left";
    }
    // future region shade (window not yet lived)
    if (d.nowX != null) { const x = X(d.nowX);
      g.fillStyle = "rgba(120,120,200,.05)"; g.fillRect(x, padT, padL + pw - x, ph);
      g.strokeStyle = "rgba(160,160,220,.35)"; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(x, padT); g.lineTo(x, padT + ph); g.stroke(); g.setLineDash([]); }
    // empirical fan cone
    if (d.fan && d.fan.length > 1) {
      g.beginPath();
      d.fan.forEach((f, i) => i ? g.lineTo(X(f[0]), Y(f[1])) : g.moveTo(X(f[0]), Y(f[1])));
      for (let i = d.fan.length - 1; i >= 0; i--) g.lineTo(X(d.fan[i][0]), Y(d.fan[i][2]));
      g.closePath(); g.fillStyle = "rgba(38,166,154,.12)"; g.fill();
      g.beginPath(); d.fan.forEach((f, i) => i ? g.lineTo(X(f[0]), Y(f[1])) : g.moveTo(X(f[0]), Y(f[1])));
      g.strokeStyle = UP; g.globalAlpha = .5; g.stroke(); g.globalAlpha = 1;
      g.beginPath(); d.fan.forEach((f, i) => i ? g.lineTo(X(f[0]), Y(f[2])) : g.moveTo(X(f[0]), Y(f[2])));
      g.strokeStyle = DN; g.stroke();
    }
    // levels: entry gold solid, TP green dashed, SL red dashed + labels
    for (const L of (d.levels || [])) { const y = Y(L.p); if (y < padT - 2 || y > padT + ph + 2) continue;
      g.strokeStyle = L.color; g.setLineDash(L.dash ? [5, 4] : []);
      g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + pw, y); g.stroke(); g.setLineDash([]);
      g.fillStyle = L.color; g.font = "bold 9px ui-monospace,monospace";
      g.fillText(`${L.tag} ${px(L.p)}`, padL + 3, y - 3); g.font = "9px ui-monospace,monospace"; }
    // the line itself: gradient area + 1.8px stroke, coloured by window move
    const up = d.pts[d.pts.length - 1].p >= d.pts[0].p, col = up ? UP : DN;
    const grad = g.createLinearGradient(0, padT, 0, padT + ph);
    grad.addColorStop(0, up ? "rgba(38,166,154,.25)" : "rgba(239,83,80,.25)"); grad.addColorStop(1, "rgba(0,0,0,0)");
    g.beginPath(); d.pts.forEach((p, i) => i ? g.lineTo(X(p.t), Y(p.p)) : g.moveTo(X(p.t), Y(p.p)));
    g.lineTo(X(d.pts[d.pts.length - 1].t), padT + ph); g.lineTo(X(d.pts[0].t), padT + ph); g.closePath();
    g.fillStyle = grad; g.fill();
    g.beginPath(); d.pts.forEach((p, i) => i ? g.lineTo(X(p.t), Y(p.p)) : g.moveTo(X(p.t), Y(p.p)));
    g.strokeStyle = col; g.lineWidth = 1.8; g.stroke(); g.lineWidth = 1;
    // last price: dot + axis chip (always visible)
    const lp = d.pts[d.pts.length - 1], lx = X(lp.t), ly = Y(lp.p);
    g.fillStyle = col; g.beginPath(); g.arc(lx, ly, 3, 0, 7); g.fill();
    if (!this.spark) { g.fillRect(padL + pw + 1, ly - 8, padR - 2, 16);
      g.fillStyle = "#0b0b14"; g.font = "bold 9px ui-monospace,monospace";
      g.fillText(px(lp.p), padL + pw + 4, ly + 3); g.font = "9px ui-monospace,monospace"; }
    // crosshair with time·price readout
    if (this.cross && !this.spark) { const { x, y } = this.cross;
      if (x >= padL && x <= padL + pw && y >= padT && y <= padT + ph) {
        g.strokeStyle = "rgba(200,200,230,.4)"; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(x, padT); g.lineTo(x, padT + ph); g.moveTo(padL, y); g.lineTo(padL + pw, y); g.stroke(); g.setLineDash([]);
        const ct = t0 + (x - padL) / pw * (t1 - t0), cp = hi - (y - padT) / ph * (hi - lo), dt = new Date(ct * 1000);
        g.fillStyle = INK; g.fillText(`${String(dt.getHours()).padStart(2, "0")}:${String(dt.getMinutes()).padStart(2, "0")}:${String(dt.getSeconds()).padStart(2, "0")} · ${px(cp)}`, Math.min(x + 6, W - 120), 12); } }
  }
}


/* ═══════════ shell: assets, tabs, focus bar ═══════════ */
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
  S.symbol = sym;
  $$(".asset-btn").forEach(b => b.classList.toggle("active", b.dataset.sym === sym));
  syncToolbar();
  refreshActiveView(true);
  buildQuads(); loadCandles();
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
    box.innerHTML = `<div class="dim pad">loading…</div>`;
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
function _drawOwn() {
  const host = $("#chart-own"); if (!host) return;
  const a = S.snap?.assets?.[S.symbol]; if (!a) return;
  const d = a.tfs?.[S.focusTf]; if (!d) return;
  const price = S.prices[S.symbol] ?? a.price; if (!price) return;
  const now = Date.now() / 1000;
  const t1 = d.cycle_end || (now + 600);
  const t0 = d.cycle_start || (now - 600);
  let past = realisedSeries(t0, now);
  if (!past.length || now - past[past.length - 1].t > 2) past.push({ t: now, p: price });
  else past[past.length - 1] = { t: now, p: price };
  if (!host._lc) { host.innerHTML = ""; host._lc = new LineChart(host); }
  const b = d.branches;
  const fan = (b?.fan && b.fan.length > 2)
    ? b.fan.map(p => [now + p.f * Math.max(1, t1 - now), p.q90, p.q10]) : null;
  const levels = [];
  if (d.entry) levels.push({ p: d.entry, tag: "ENTRY", color: "#c9a227", dash: false });
  if (d.tp) levels.push({ p: d.tp, tag: "TP", color: "#26a69a", dash: true });
  if (d.sl) levels.push({ p: d.sl, tag: "SL", color: "#ef5350", dash: true });
  host._lc.set({ pts: past, t0, t1, nowX: now, fan, levels });
}

/* ═══════ BINANCE chart: the chart you see on binance.com — Binance
   klines (candles + volume + MA7/MA25) from our Binance feed, drawn
   with the vendored lightweight-charts engine. No iframe, no fake. ═══ */
const Bn = { main: null, quads: {}, key: "" };
function bnMake(el, mini) {
  const LW = window.LightweightCharts; if (!LW || !el) return null;
  const chart = LW.createChart(el, {
    layout: { background: { color: "transparent" }, textColor: "#8a8aa0", fontSize: 9 },
    grid: { vertLines: { color: "rgba(140,140,170,.08)" }, horzLines: { color: "rgba(140,140,170,.08)" } },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: "rgba(140,140,170,.25)" },
    width: el.clientWidth || 300, height: el.clientHeight || (mini ? 142 : 340),
  });
  const candles = chart.addCandlestickSeries({ upColor: "#26a69a", downColor: "#ef5350",
    borderVisible: false, wickUpColor: "#26a69a", wickDownColor: "#ef5350" });
  const vol = chart.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "volume" } });
  chart.priceScale("").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
  const ma = (color) => chart.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
  const out = { chart, candles, vol, ma7: ma("rgba(240,185,11,.85)"), ma25: ma("rgba(200,120,255,.7)"), last: null };
  new ResizeObserver(() => { try { chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }); } catch {} }).observe(el);
  return out;
}
function bnFeed(inst, bars) {
  if (!inst || !bars.length) return;
  const cs = bars.map(b => ({ time: b.t, open: b.o, high: b.h, low: b.l, close: b.c }));
  inst.candles.setData(cs);
  inst.vol.setData(bars.map(b => ({ time: b.t, value: b.v || 0,
    color: b.c >= b.o ? "rgba(38,166,154,.4)" : "rgba(239,83,80,.4)" })));
  const maD = (n) => cs.map((c, i) => i < n - 1 ? null
    : { time: c.time, value: cs.slice(i - n + 1, i + 1).reduce((x, y) => x + y.close, 0) / n }).filter(Boolean);
  inst.ma7.setData(maD(7)); inst.ma25.setData(maD(25));
  const lb = bars[bars.length - 1];
  inst.last = { time: lb.t, open: lb.o, high: lb.h, low: lb.l, close: lb.c, v: lb.v || 0 };
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
  if (mainEl && !Bn.main) { Bn.main = bnMake(mainEl, false); }
  if (Bn.main) bnLoad(S.focusTf, Bn.main, mainEl);
  for (const tf of TF) {
    const el = $("#qbn-" + tf); if (!el) continue;
    if (!Bn.quads[tf]) Bn.quads[tf] = bnMake(el, true);
    if (Bn.quads[tf]) bnLoad(tf, Bn.quads[tf], el);
  }
}
function bnTickMain() {
  const inst = Bn.main, p = S.prices?.[S.symbol];
  if (!inst || !inst.last || !(p > 0)) return;
  const l = inst.last;
  l.close = p; l.high = Math.max(l.high, p); l.low = Math.min(l.low, p);
  try { inst.candles.update({ time: l.time, open: l.open, high: l.high, low: l.low, close: l.close }); } catch {}
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
  $("#branch-scope").textContent = `${S.symbol} · ${S.focusTf.toUpperCase()} · ${b?.method === "empirical-first-touch" ? `measured first-touch · n=${b.n}` : "approx (thin tape)"}`;
  if (!b || !a) {
    stats.innerHTML = "";
    cvHost.innerHTML = `<div class="dim pad" style="font-size:11px">waiting for a direction + TP/SL on ${S.focusTf.toUpperCase()}…</div>`;
    $("#branch-note").textContent =
      "bands appear once the window has a direction and a seeded stop";
    return;
  }
  const _pct = p => p > 0 && p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`;
  stats.innerHTML = `
    <div class="kv"><span>P(target before stop)</span><b class="acc-up">${_pct(b.p_target)}</b></div>
    <div class="kv"><span>P(stop first)</span><b class="acc-dn">${_pct(b.p_stop)}</b></div>
    <div class="kv"><span>P(close between)</span><b>${_pct(b.p_between)}</b></div>
    <div class="kv"><span>P(close down)</span><b>${Math.round(b.p_close_down * 100)}%</b></div>
    <div class="kv"><span>σ (remaining)</span><b>${px(b.sigma)}</b></div>
    <div class="kv"><span>expectancy (barrier pair)</span><b class="${(b.ev_bps ?? 0) >= 0 ? "acc-up" : "acc-dn"}">${b.ev_bps != null ? `${b.ev_bps >= 0 ? "+" : ""}${fmt(b.ev_bps, 1)} bp` : "—"}</b></div>
    <div class="kv"><span>P(target) 95% interval</span><b>${b.ci_target ? `${Math.round(b.ci_target[0] * 100)}–${Math.round(b.ci_target[1] * 100)}%` : "—"}</b></div>
    <div class="kv"><span>forecast audit (all windows)</span><b>${S.snap?.prob_audit?.n ? `n=${S.snap.prob_audit.n} · pred ${Math.round(S.snap.prob_audit.mean_predicted * 100)}% vs realised ${Math.round(S.snap.prob_audit.realised_tp_first * 100)}% · ${esc(S.snap.prob_audit.verdict || "")}` : "grading after settled windows"}</b></div>
    <div class="kv"><span>basis</span><b>${b.method === "empirical-first-touch" ? `measured · n=${b.n} windows` : "approx · thin tape"}</b></div>
    <div class="kv"><span>time left</span><b>${Math.round(b.rem_s / 60)}m ${Math.round(b.rem_s % 60)}s</b></div>`;
  $("#branch-note").textContent = b.method === "empirical-first-touch"
    ? `measured: first-touch outcomes replayed over ${b.n} historical windows on this asset's own tape (both-barrier bars count as stop-first, ${b.ambig} ambiguous) · bands = empirical return quantiles · realised path in white`
    : "thin tape — driftless approximation until 20+ windows can be replayed · realised path in white";
  // ---- canvas ----
  const w = cvHost.clientWidth || 300, h = cvHost.clientHeight || 170;
  let cv = cvHost.querySelector("canvas");
  if (!cv) {
    cvHost.innerHTML = "";
    cv = document.createElement("canvas");
    cv.style.cssText = "width:100%;height:100%;display:block";
    cvHost.appendChild(cv);
  }
  cv.width = w * 2; cv.height = h * 2;
  const g = cv.getContext("2d");
  g.setTransform(2, 0, 0, 2, 0, 0);
  g.clearRect(0, 0, w, h);
  const t0 = d.cycle_start || (Date.now() / 1000 - 60);
  const t1 = d.cycle_end || (Date.now() / 1000 + b.rem_s);
  // the fan is anchored at NOW (bands are quantiles of the REMAINING
  // window from the current price) — the old code spread them across
  // the whole cycle, so the cone floated away from the realised path
  const tN = b.now || (Date.now() / 1000);
  const levels = [b.entry, b.tp, b.sl, ...b.bands.map(x => x.level)];
  let lo = Math.min(...levels), hi = Math.max(...levels);
  const pad = (hi - lo) * 0.12 || Math.abs(b.entry * 1e-3) || 1;
  lo -= pad; hi += pad;
  const X = t => ((t - t0) / Math.max(1, t1 - t0)) * (w - 42) + 1;
  const Y = p2 => h - 14 - ((p2 - lo) / (hi - lo || 1)) * (h - 20);
  const XF = f => X(tN + f * Math.max(1, t1 - tN));   // fan x at fraction f
  // shaded EMPIRICAL cone (quantiles per elapsed fraction, measured
  // on this asset's tape), rooted at (now, current price)
  if (b.fan && b.fan.length > 2) {
    for (const [hiK, loK, fill] of [["q95", "q05", "rgba(0,212,240,.08)"], ["q75", "q25", "rgba(0,212,240,.14)"]]) {
      g.beginPath();
      g.moveTo(X(tN), Y(b.entry));
      b.fan.forEach(p => g.lineTo(XF(p.f), Y(p[hiK])));
      for (let i = b.fan.length - 1; i >= 0; i--) g.lineTo(XF(b.fan[i].f), Y(b.fan[i][loK]));
      g.closePath(); g.fillStyle = fill; g.fill();
    }
    g.beginPath(); g.moveTo(X(tN), Y(b.entry));
    b.fan.forEach(p => g.lineTo(XF(p.f), Y(p.q50)));
    g.strokeStyle = "rgba(216,216,232,.5)"; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]);
  } else {
    const upper = b.bands.filter(x => x.f >= 0.5);
    const lower = b.bands.filter(x => x.f <= 0.5);
    g.beginPath();
    g.moveTo(X(tN), Y(b.entry));
    upper.forEach(x => g.lineTo(XF(x.f), Y(x.level)));
    for (let i = lower.length - 1; i >= 0; i--) g.lineTo(XF(lower[i].f), Y(lower[i].level));
    g.closePath();
    g.fillStyle = "rgba(240,56,90,.08)"; g.fill();
  }
  // median band line
  const med = b.bands.find(x => x.f === 0.5);
  if (med) {
    g.beginPath(); g.moveTo(X(tN), Y(b.entry)); g.lineTo(XF(med.f), Y(med.level));
    g.strokeStyle = "rgba(216,216,232,.5)"; g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
  }
  // TP / SL / entry — TP and SL are FUTURE barriers: draw them from
  // the now-line to the window end, not across the realised past
  const hline = (lv, color, label, dash, fromX) => {
    g.beginPath(); g.moveTo(fromX ?? X(t0), Y(lv)); g.lineTo(X(t1), Y(lv));
    g.strokeStyle = color; g.lineWidth = 1.2; g.setLineDash(dash || [6, 4]); g.stroke(); g.setLineDash([]);
    g.fillStyle = color; g.font = "9px monospace"; g.textAlign = "left";
    g.fillText(`${label} ${px(lv)}`, (fromX ?? X(t0)) + 3, Y(lv) - 3);
  };
  hline(b.tp, "#22c98a", "TP →", [6, 4], X(tN));
  hline(b.sl, "#f0385a", "SL →", [6, 4], X(tN));
  hline(b.entry, "#d8d8e8", "now", [2, 3]);
  // realised path (white) — candle series from the window start,
  // not just ticks since page load (was starting mid-chart)
  const hist = realisedSeries(t0 - 1, Date.now() / 1000);
  if (hist.length > 1) {
    g.beginPath();
    hist.forEach((pp, i) => { const xx = X(pp.t), yy = Y(pp.p); i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); });
    g.strokeStyle = "#ffffff"; g.lineWidth = 1.3; g.stroke();
  }
  // now marker
  const nowX = X(Math.min(Date.now() / 1000, t1));
  g.beginPath(); g.moveTo(nowX, 4); g.lineTo(nowX, h - 12);
  g.strokeStyle = "rgba(201,162,39,.7)"; g.lineWidth = 1; g.stroke();
  g.fillStyle = "#c9a227"; g.font = "9px monospace"; g.textAlign = "right";
  g.fillText("now", nowX - 3, 12);
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
  S.focusTf = tf;
  // choosing a timeframe switches the DASHBOARD to focus mode:
  // that window + its sub-timeframe TPs only (⊞ all timeframes reverts)
  S.tfFocus = true;
  try { localStorage.setItem("tfFocus", "1"); } catch (e) { /* private mode */ }
  syncToolbar(); renderWhy(); renderQuads(); S.lastViewFetch = 0;
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

function renderHero() {
  const tf = S.focusTf;
  const a = S.snap?.assets?.[S.symbol] || {};
  const cyc = a.tfs?.[tf];
  const res = S.snap?.systems?.[`${S.symbol}|${tf}`] || {};
  const m = TF_META[tf];
  const dir = cyc?.direction ?? res.direction ?? 0;
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
    badge.innerHTML = `${dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO CALL"}${weak && dir ? "*" : ""}<small>${S.symbol} · ${tf.toUpperCase()} ${m.mins}m · conf ${fx(res.confidence ?? 0, 3)}${cyc?.issued ? " · ISSUED" : " · no trade"}</small>`;
  }
  const stats = $("#hero-stats");
  if (stats) {
    stats.innerHTML = `
      <div class="kv"><span>call</span><b class="${dir > 0 ? (weak ? "flat" : "up") : dir < 0 ? (weak ? "flat" : "down") : "flat"}">${dir > 0 ? "▲ BULL" : dir < 0 ? "▼ BEAR" : "● FLAT"}${weak && dir ? "*" : ""}</b></div>
      <div class="kv"><span>confidence</span><b>${fx(res.confidence ?? 0, 3)} ${cyc?.issued ? "· ISSUED" : "· below floor"}</b></div>
      <div class="kv"><span>cci adj</span><b>${fx(res.cci_adjusted ?? 0, 4)}</b></div>
      <div class="kv"><span>countdown</span><b>${cyc?.countdown_s != null ? `${Math.floor(cyc.countdown_s / 60)}m ${cyc.countdown_s % 60}s` : "—"}</b></div>
      <div class="kv"><span>entry (locked)</span><b>${cyc?.entry ? px(cyc.entry) : (res.lock?.price ? px(res.lock.price) : "—")}</b></div>
      <div class="kv"><span>price now</span><b>${a.price ? px(a.price) : "—"}</b></div>
      <div class="kv"><span>stop loss</span><b class="acc-dn">${cyc?.sl ? px(cyc.sl) : "—"}${cyc?.sl_provisional ? ' <span class="dim" style="font-size:9px">prov</span>' : ""}</b></div>
      <div class="kv"><span>ratio</span><b>${cyc?.ratio ? "1:" + cyc.ratio : "—"}</b></div>`;
  }
  // TP ladder — the TF's own horizon ALWAYS first (user rule)
  const tg = cyc?.targets || [];
  const order = tg.map((_, i) => i);
  if (tg.length >= 4) order.unshift(order.pop());   // [3,0,1,2]
  const strip = $("#subtf-strip");
  if (strip) {
    strip.innerHTML = order.map(i => {
      const t = tg[i];
      if (!t) return "";
      const isTf = i === tg.length - 1;
      const st = t.status === "hit" ? "HIT ✓" : t.status === "locked" ? "LOCK 🔒" : "open";
      const cls = t.status === "hit" ? "acc-up" : t.status === "locked" ? "acc-warn" : "dim";
      const _entry = cyc?.entry || res.lock?.price || a.price || 0;
      const _rel = (t.level && _entry) ? ` <span class="dim">${((t.level / _entry - 1) * 100).toFixed(2)}% from entry</span>` : "";
      return `<div class="kv"><span>${isTf ? `TF ${t.minutes}m — this window` : `TP${i + 1} · ${t.minutes}m sub-frame`}</span><b class="${cls}">${t.level ? px(t.level) : "—"} <span class="dim">${st}</span>${_rel}</b></div>`;
    }).join("") || `<div class="dim pad">waiting for the first lock…</div>`;
  }
  const rn = $("#risk-note");
  if (rn) {
    const rk = cyc?.risk || {};
    rn.textContent = (rk.note
      ? `TP/SL from measured moves — ${rk.note}${rk.provisional ? " · levels provisional" : ""} · frozen at the lock; an emergency re-lock re-anchors the same bps at the new entry`
      : "TP/SL: measured-moves block pending (needs a locked cycle)")
      + (res.asset_nature ? ` · ${S.symbol} profile: ${res.asset_nature} · issue floor ${fmt(res.asset_floor, 3)}` : "")
      + (res.behavior?.action ? ` · behaviour: ${res.behavior.action}` : "");
  }
  // chart overlays: SL, TP ladder (horizon first), liquidation clusters
  const lines = [];
  if (cyc?.sl) lines.push({ price: cyc.sl, color: "#f0385a", title: "SL" });
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
  order.forEach(i => {
    const t = tg[i];
    if (!t || !t.level) return;
    const isTf = i === tg.length - 1;
    lines.push({
      price: t.level,
      color: t.status === "hit" ? "#22c98a" : t.status === "locked" ? "#c9a227" : "#00d4f0",
      title: isTf ? `TF${t.minutes}m` : `TP${i + 1}${t.status === "hit" ? " ✓" : t.status === "locked" ? " 🔒" : ""}`,
      dash: t.status !== "locked",
    });
  });
  S.charts.hero?.setLines(lines);
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
  else if (S.view === "history") loadHistory();
  else if (S.view === "settings") loadSettings();
}

/* ═══════════ dashboard: quadrants ═══════════ */
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
      <div class="q-spark" id="qs-${tf}"></div>
      <div class="q-price" id="qp-${tf}">—</div>
      <div class="q-dir flat" id="qd-${tf}">—</div>
      <div class="q-tps" id="qt-${tf}"></div>
      <div class="q-foot"><span id="qf-${tf}"></span><span id="qa-${tf}"></span></div>`;
    grid.appendChild(q);
    q.addEventListener("click", () => { if (S.focusTf !== tf) setFocusTf(tf); });
    S.charts[tf] = new LineChart($("#qc-" + tf));
    S.spark[tf] = new LineChart($("#qs-" + tf), { spark: true });
  });
  // main OWN chart is the LineChart-from-now renderer (drawMainChart),
  // (LineChart v2 line mode — candles crushed the live action)
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
    const dir = cyc?.direction ?? res?.direction ?? 0;
    const qd = $("#qd-" + tf);
    if (qd) {
      qd.className = "q-dir " + (dir > 0 ? "up" : dir < 0 ? "down" : "flat");
      qd.innerHTML = `${dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO TRADE"}
        <span class="iss">${cyc?.issued ? "ISSUED · targets active" : "below floor · no trade"}</span>`;
    }
    $("#qf-" + tf).textContent = res ? `CCI ${fmt(res.cci_adjusted, 4)} · conf ${fmt(res.confidence, 3)}` : "";
    const ag = res?.agent;
    $("#qa-" + tf).textContent = ag && ag.source !== "none"
      ? `AI ${ag.direction > 0 ? "+" : ag.direction < 0 ? "−" : "0"} (${ag.source})` : "AI —";
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
    // TP/SL overlay lines on the chart
    const lines = [];
    if (cyc?.sl) lines.push({ price: cyc.sl, color: "#f0385a", title: "SL" });
    // liquidation clusters (measured @forceOrder buckets) as purple overlays
    // (liquidation-cluster overlays live on the hero chart only — on
    // the small quad cards they were two more colliding labels)
    // quad cards show ONLY the horizon TP + SL lines: every sub-target
    // line on a 90 px card was an unreadable label stack (the values
    // themselves are in the text row under the card and the calls table)
    const _hz = (cyc?.targets || [])[(cyc?.targets || []).length - 1];
    if (_hz?.level) lines.push({
      price: _hz.level, color: "#c9a227",
      title: `TP${_hz.minutes}m`, dash: false,
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
    const on = st?.settings?.auto_predict !== false;
    S.autoPredict = on;
    const chip = $("#auto-chip");
    if (chip) {
      chip.textContent = on ? "AUTO: ON" : "AUTO: OFF";
      chip.className = "btn-analyse btn-auto " + (on ? "auto-on" : "auto-off");
      chip.title = on
        ? "AUTO-PREDICT on — analysis re-runs by itself every ~45 s; every tab is always fresh. Click for MANUAL mode (▶ ANALYSE only)."
        : "MANUAL mode — press ▶ ANALYSE on the dashboard for a fresh full analysis. Click for AUTO.";
    }
    const row = $("#st-auto");
    if (row) row.textContent = on ? "ON — analysis runs itself (click to toggle)" : "OFF — manual ▶ ANALYSE only (click to toggle)";
  } catch { /* offline */ }
}
function bindAutoChip() {
  const chip = $("#auto-chip");
  if (chip) chip.addEventListener("click", async () => {
    const next = !(S.autoPredict !== false);
    try {
      await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" },
                                   body: JSON.stringify({ auto_predict: next }) });
      toast(next ? "AUTO-PREDICT on — analysis runs itself" : "MANUAL — press ▶ ANALYSE for a fresh run", "ok");
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
  $$("#seg-chart .seg-btn").forEach(b =>
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
  if (mode === "tv" && wrap) fill(wrap, S.focusTf);
  if (mode === "binance") bnEnsure();
  if (mode === "own") drawMainChart();
  // every sub-timeframe card follows the same toggle: our LineChart in
  // OWN, Binance candles in BINANCE, TradingView widget in TV mode.
  for (const tf of TF) {
    const o = $("#qc-" + tf), sp = $("#qs-" + tf),
          slot = $("#qtv-" + tf), bslot = $("#qbn-" + tf);
    if (o) o.style.display = mode === "own" ? "" : "none";
    if (sp) sp.style.display = mode === "own" ? "" : "none";
    if (slot) { slot.classList.toggle("hidden", mode !== "tv"); if (mode === "tv") fill(slot, tf); }
    if (bslot) bslot.classList.toggle("hidden", mode !== "binance");
  }
}
function bindChartSrc() {
  try { S.chartSrc = localStorage.getItem("chartSrc") || "binance"; } catch { S.chartSrc = "binance"; }
  if (!["binance", "tv", "own"].includes(S.chartSrc)) S.chartSrc = "binance";
  $$("#seg-chart .seg-btn").forEach(b => b.addEventListener("click", () => {
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
      const r = await api(`/api/analyse/${sym}?tf=${S.focusTf}`, { method: "POST" });
      if (!r.ok) { toast(r.error || "analysis unavailable", "err"); b.disabled = false; return; }
      // show the panel on the dashboard immediately — timestamps anchor
      // to the button press (started_at comes from the server)
      setView("dashboard");
      const panel = $("#analyse-panel");
      panel?.classList.remove("hidden");
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
    } catch (e) { toast("analyse: " + e.message, "err"); }
    b.disabled = false;
  });
}
function onAnalyseStage(msg) {
  const st = msg.stage; if (!st) return;
  const row = $("#anr-" + st.stage);
  if (!row) return;
  row.className = "an-row " + (st.status || "done");
  $(".an-ms", row).textContent = st.duration_ms + " ms";
  $(".an-at", row).textContent = new Date(st.t_start_ms).toISOString().slice(11, 23) + "Z";
  $(".an-sum", row).textContent = st.summary || "";
  row.title = st.summary || "";
}
function onAnalyseDone(msg) {
  const b = $("#btn-analyse"); if (b) b.disabled = false;
  const r = msg.result;
  const sub = $("#an-sub");
  if (!r) {
    if (sub) sub.textContent = "FAILED — " + (msg.error || "unknown error");
    toast("analysis failed: " + (msg.error || ""), "err");
    return;
  }
  if (sub) sub.textContent =
    `finished ${new Date(r.calculation_period.finished_at * 1000).toISOString().slice(11, 23)}Z · total ${r.elapsed_ms} ms`;
  const box = $("#an-result");
  if (!box) return;
  box.classList.remove("hidden");
  const dirCls = r.signal === "BUY" ? "buy" : r.signal === "SELL" ? "sell" : "flat";
  const ai = r.ai || {};
  let aiLine = ai.available
    ? Object.entries(ai.reviews || {}).map(([name, rv]) =>
        `${name}: ${rv.ok ? ((rv.errors || []).length ? (rv.errors || []).length + " issue(s) found — " + (rv.reason || "") : "no issues — " + (rv.reason || "concurs")) : "unavailable (" + (rv.error || "") + ")"}`).join(" · ")
    : (ai.note || "keyless — deterministic pipeline only");
  if (ai.correction) {
    aiLine += ` · ⚠ AI CORRECTION by ${ai.correction.by}: ${ai.correction.direction > 0 ? "BUY" : "SELL"} (${Math.round(ai.correction.confidence * 100)}%) — ${ai.correction.reason}`;
  }
  const period = (r.calculation_period?.stages || []).map(c =>
    `${c.stage} ${Math.round(c.ms)}ms`).join(" → ");
  box.innerHTML = `<div class="an-box">
    <div class="an-signal ${dirCls}">
      <div class="big">${r.signal}</div>
      <div class="conf">confidence ${fx(r.confidence, 3)}${r.issued ? " · ISSUED" : " · below floor"}</div>
      <div class="conf">entry ${px(r.price)} · ATR ${fmt(r.atr, 4)}</div>
    </div>
    <div class="an-grid">
      ${r.tps.map(t => `<div class="an-cell"><div class="k">TP${t.k} · ${t.minutes}m</div><div class="v">${px(t.level)}</div></div>`).join("")}
      <div class="an-cell"><div class="k">STOP LOSS</div><div class="v" style="color:#f0385a">${r.sl ? px(r.sl) : "—"}</div></div>
      <div class="an-cell"><div class="k">LADDER</div><div class="v" style="font-size:10.5px;font-weight:400">${esc(r.ladder?.source || "—")}</div></div>
      <div class="an-w">weightage for ${msg.symbol} (per-asset mix): ${(r.weightage || []).map(w => `F${w.id} <b>${w.w}</b>`).join(" · ")}<br/><span class="dim">${esc(r.weightage_note || "")}</span></div>
      <div class="an-ai">AI review — ${esc(aiLine)}</div>
      <div class="an-period">calculation period: ${esc(period)} · total ${r.elapsed_ms} ms</div>
    </div></div>`;
  // SYNC: the analysis recomputes every subsystem — refetch whatever view
  // is open so formulas/emotion/physics/brain/news all show the same run
  refreshActiveView(true);
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
  $("#feed-mode").textContent = a[BTC_MODE()]?.mode === "live" ? "LIVE (Binance)" : "SIM (regime-switching)";
  const led = (id, on, warn) => {
    const e = $(id); if (e) e.className = "led " + (warn ? "warn" : on ? "on" : "off");
  };
  led("#led-stream", S.lastMsgAt && Date.now() - S.lastMsgAt < 5000);
  led("#led-learn", true, S.snap?.learning?.high_alert);
  // brain connection — same component the brain tab shows, mirrored here
  // so the dashboard can never say "nothing" while the brain is green
  try {
    const fly = S.snap?.systems?.[`${S.symbol || "BTC"}|${S.focusTf || "tf1"}`]?.brain?.flywire;
    const bc = fly?.chip
      || (fly?.ready ? { kind: "ok", text: "FlyWire v783 · live" }
        : { kind: "absent", text: "brain — " + (fly?.reason || "absent") });
    led("#led-brain", bc.kind === "ok", bc.kind === "build");
    if (bc.kind === "absent") led("#led-brain", false, false);
    const bchip = $("#brain-chip");
    if (bchip) {
      bchip.textContent = "🧠 " + bc.text;
      const cc = bc.kind === "ok" ? "var(--green)"
        : bc.kind === "build" ? "var(--gold)" : "var(--red)";
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
  const dir = s.direction ?? 0;
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
    ? "call: NO TRADE until signals align"
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
  el.innerHTML = lockBlock + ladderBlock + sparkHtml + ticksBlock + whoBlock + convBlock + exBlock + `
    <div class="why-head">
      <div class="why-dir ${dir > 0 ? "up" : dir < 0 ? "down" : "flat"}">
        ${dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO TRADE"}<span class="sym">${S.symbol} · ${tf.toUpperCase()} · ${TF_META[tf].mins}m</span></div>
      <div class="why-conf">CONFIDENCE
        <div class="bar"><i class="${s.confidence > 0.35 ? "pos" : "neg"}" style="width:${clamp((s.confidence || 0) * 100, 0, 100)}%"></i></div>
        <b>${fx(s.confidence, 3)}</b> / floor ${bd.floor ?? 0.35}</div>
      <div class="why-badge ${s.issued ? "ok" : ""}">${s.issued ? "✓ SIGNAL ISSUED" : "NO TRADE · below floor"}</div>
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
        <h5>CONFIDENCE CHAIN — spec order</h5>
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
            ? `<div class="kv" style="margin-top:6px"><span title="${esc(tip)}">brain +${fmt(f.vote, 3)} <span class="dim">(FlyWire v783 · conf ${fmt(f.conf, 2)}${tip ? " · hover for top channels" : ""})</span></span><b class="${f.vote >= 0 ? "up" : "down"}">${f.vote >= 0 ? "approach" : "avoid"}</b></div>`
            : `<div class="kv" style="margin-top:6px"><span>brain</span><b class="down" title="${esc(f.reason || "")}">absent (${esc(f.reason || "vote withheld")}) — weight 0</b></div>`;
        })()}
        <div class="hint">4-vote double-check: PCV + agent + brain (+ emotion/physics here) must agree to issue.</div>
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
      <td class="mono">${st.n}</td>
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
      <div class="kv"><span>predictability</span><b>${bits != null ? bits.toFixed(3) + " bit" : (gauge?.note || "needs 40 windows")}</b></div>
      <div class="kv"><span>calib a · b</span><b>${ev.calib_a ?? "—"} · ${ev.b ?? "—"}</b></div>
    </div>
    <div class="hint">${esc(rec.message || "")} · ${esc(m.expert?.message || "")}</div>
    ${gauge && gauge.ready ? `<div class="hint ${gauge.bits < 0.02 ? "chip-warn" : "chip-ok"}">P(up)=${gauge.p_up} · H₂=${gauge.h2_bits} bit — ${esc(gauge.note)}</div>` : ""}
    <table class="tbl" style="margin-top:6px">
      <thead><tr><th>SOURCE</th><th>slope aᵢ</th><th>MI (bits)</th><th>coefficient</th></tr></thead>
      <tbody>${Object.entries(ev.slopes || {}).map(([k, v]) => `<tr>
        <td>${esc(k)}</td><td>${v}</td><td>${(ev.mi_bits || {})[k] ?? "—"}</td>
        <td>${(m.coefficients || {})[k] ?? "—"}</td></tr>`).join("")}</tbody>
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
      ? `<br/><span class="dim" style="font-size:9.5px">* weak lean inside noise floor — never issued</span>` : "";
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
    const sl = cyc?.sl ? `${px(cyc.sl)}${cyc.sl_provisional ? ' <span class="dim" style="font-size:9px">prov</span>' : ""}` : "—";
    return `<tr class="${tf === S.focusTf ? "focus-row" : ""}">
      <td class="mono">${tf.toUpperCase()}</td>
      <td class="mono">${m.mins}m</td>
      <td>${dirTxt} <span class="dim">${cyc?.issued ? "· issued" : "· no trade"}</span>${_leanNote}</td>
      <td class="mono">${fmt(res?.confidence, 3)}</td>
      ${tps}
      <td class="mono sl-cell">${sl}</td>
      <td class="mono">${cyc?.ratio ? "1:" + cyc.ratio : "—"}</td>
    </tr>`;
  }).join("");
  el.innerHTML = `<table class="tbl calls-tbl">
    <thead><tr><th>TF</th><th>WINDOW</th><th>CALL</th><th>CONF</th>
      <th>TF-TP (horizon)</th><th>TP1</th><th>TP2</th><th>TP3</th><th>SL</th><th>RATIO</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="hint">The TF's own horizon TP (the window itself) is always calculated first and shown first; TP1–TP3 are its sub-timeframes (one stop-loss, one ratio). Levels come from the engine's ATR projection at every GRC.${S.tfFocus ? " Focused: only " + S.focusTf.toUpperCase() + " shown — use ⊞ all timeframes." : ""}</div>`;
}

/* ═══════════ candles (per symbol, per TF) ═══════════ */
async function loadCandles() {
  try {
    // ONE dense minute-series fetch drives every TF card chart (the
    // aggregated 44/130/247-minute bars were only 4-5 points → smooth
    // fake curves; traders want the real zigzag). TP/lock overlays stay
    // per-TF below.
    const md = await api(`/api/candles?symbol=${S.symbol}&tf=tf1&n=260`);
    const mbars = (md.bars || []).map(x =>
      ({ time: x.t, open: x.o, high: x.h, low: x.l, close: x.c }));
    await Promise.all(TF.map(async tf => {
      const d = { bars: md.bars };
      const bars = d.bars || [];
      const c = S.charts[tf]; if (!c || !bars.length) return;
      const markers = [];
      for (let i = 2; i < bars.length - 2; i++) {
        const b = bars[i];
        if (b.h > bars[i-1].h && b.h > bars[i-2].h && b.h > bars[i+1].h && b.h > bars[i+2].h)
          markers.push({ time: b.t, above: true, color: "#f0385a", glyph: "◆" });
        if (b.l < bars[i-1].l && b.l < bars[i-2].l && b.l < bars[i+1].l && b.l < bars[i+2].l)
          markers.push({ time: b.t, above: false, color: "#00d4f0", glyph: "◆" });
      }
      const cyc = S.snap?.assets?.[S.symbol]?.tfs?.[tf];
      (cyc?.targets || []).forEach(t => {
        if (t.status === "hit" && t.hit_ts)
          markers.push({ time: Math.floor(t.hit_ts / 60) * 60, above: true, color: "#22c98a", glyph: "▲" });
      });
      const mapped = bars.map(x => ({ time: x.t, open: x.o, high: x.h, low: x.l, close: x.c }));
      c.setData(mapped);
      c.setMarkers(markers.slice(-36));
      // seed the realised-path history (hero line chart + branches):
      // without this the white path only existed from page load and
      // the forward fan floated with nothing behind it
      if (tf === S.focusTf) {
        const _norm = t => t > 1e12 ? t / 1000 : t;
        const seeded = mapped.map(x => ({ t: _norm(x.time), p: x.close }))
          .filter(p2 => p2.p > 0);
        const tail = (S.priceHist || [])
          .filter(p2 => p2.t > (seeded[seeded.length - 1]?.t || 0));
        S.priceHist = [...seeded, ...tail].slice(-900);
        // full-window candle series for the charts' realised line:
        // covers the window from its start (REST backfill), so the
        // line begins at the window's left edge on a true time axis
        S.candleSeries = { symbol: S.symbol, bars: seeded };
      }
      S.spark[tf]?.setData(bars.slice(-90).map(x => ({ time: x.t, open: 0, high: x.v || 0, low: 0, close: x.v || 0 })));
    }));
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
      <div class="kv"><span>issued</span><b>${s.issued ? "YES" : "no"}</b></div>`;
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
          const tds = ["tf1", "tf2", "tf3", "tf4"].map(tf => {
            const cell = (fwd.buckets?.[`F${fid} ${S.symbol} ${tf}`] || [])
              .find(b => Math.abs(b.bucket - (Math.max(-1, Math.min(1, core[fid])) )) < 0.26);
            return cell && cell.settled
              ? `<td title="bucket ${cell.bucket}, n=${cell.n}">${(cell.p_up * 100).toFixed(0)}%<span class="dim"> n${cell.n}</span></td>`
              : `<td class="dim">—</td>`;
          }).join("");
          const cur = (fv.raw?.[fid] ?? core[fid]);
          const tOn = contribs[fid] ? Math.abs(contribs[fid].contribution) >= 1e-5 : null;
          const tk = tOn === null ? "–" : tOn ? "✓" : "✗";
          return `<tr><td title="${tOn ? "contributed this window" : tOn === null ? "no reading" : "did not contribute"}">${tk} F${fid} ${(fv.names?.[fid] || "")}</td><td>${fx(cur, 3)}</td>${tds}</tr>`;
        }).join("");
        fwEl.innerHTML = `<table class="fwd-table"><thead><tr><th>formula ✓/✗</th><th>now</th>
          <th>tf1 P(up)</th><th>tf2 P(up)</th><th>tf3 P(up)</th><th>tf4 P(up)</th></tr></thead>
          <tbody>${rows}</tbody></table>
          <div class="hint">Forward P(up) is settled strict-sign from this ledger — n≥30 per bucket before it speaks (advice §2.7). Current value beside it: superposition (§2.8).</div>`;
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
    } catch (e) { console.warn("genesis:", e); }
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
      mn = sorted[Math.floor(sorted.length * 0.05)];
      mx = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)];
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
      if (dom && bi) {
        const pct = Math.round(best * 100);
        dom.innerHTML = `
          <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap">
            <div style="font:700 26px/1 var(--mono);color:var(--gold)">${esc((typeof EMO !== "undefined" && EMO[bi - 1]) || ("E" + bi))}</div>
            <div class="dim">E${bi} · <b style="color:var(--text)">${pct}%</b> intensity · held ${heldS}s · runner-up ${esc((typeof EMO !== "undefined" && EMO[ri - 1]) || ("E" + ri))} ${Math.round(second * 100)}%${em.dominant?.tied ? ` · <span class="acc-warn">+${em.dominant.tied} channel${em.dominant.tied > 1 ? "s" : ""} tied at this extreme (saturated tape — no single winner)</span>` : ""}</div>
          </div>
          <div class="bar" style="margin-top:8px;max-width:420px"><i class="pos" style="width:${clamp(pct, 2, 100)}%"></i></div>
          <div class="hint">confidence modifier only — never the direction (advice §5.1)</div>`;
      }
    } catch (e) { /* dominant header optional */ }
    $("#em-scope").textContent = `${S.symbol} · ${S.focusTf.toUpperCase()}`;
    $("#em-narrative").innerHTML = `
      <div class="why-reason">${em.narrative || "waiting for the first emotion pass…"}</div>
      <div class="kv-row">
        <div class="kv"><span>CERC composite (E22)</span><b>${fmt(em.cerc, 4)}</b></div>
        <div class="kv"><span>aggregate</span><b>${fmt(em.aggregate, 4)}</b></div>
        <div class="kv"><span>RTP trauma</span><b>${fmt(em.rtp, 3)}${(em.rtp || 0) > 0.6 ? " ⚠" : ""}</b></div>
        <div class="kv"><span>TEDR → P6</span><b>${fmt(em.tedr, 4)}</b></div>
      </div>
      ${em.features ? `<div class="kv-row"><div class="kv"><span>measured features (tape only — never the signal, G-3)</span>
        <b>samples ${em.features.samples ?? "—"} · vol-z ${fmt(em.features.vol_z, 3)} · climax ×${fmt(em.features.climax_ratio, 2)} · violence gate ${fmt(em.features.violence_gate, 2)} · streak ${em.features.window_streak ?? "—"}</b></div></div>` : ""}
      <div class="hint">22 spec emotion formulas feed E22 CERC → confidence modulator ±15 % (never the direction, advice §5.1). Circles: conic ring = |value|, border = direction.</div>
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
    for (let i = 1; i <= 22; i++) {
      const v = em.emotions?.[i] ?? 0;
      const dir = em.directions?.[i] ?? 0;
      const cell = $("#ec-" + i);
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
      const ring = $(".em-ring", cell);
      ring.style.setProperty("--p", `${clamp(Math.abs(v) * 100, 0, 100)}%`);
      $("i", ring).textContent = (v >= 0 ? "+" : "") +
        (v === 0 ? "0.00" : fx(v, 2));   //0.004 must print0.004, not 0.00
      const tag = $(".em-tag", cell);
      let label = "NEUTRAL";
      if (Math.abs(v) > 0.7) label = "EXTREME"; else if (Math.abs(v) > 0.45) label = "ELEVATED";
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
    const bgn = S.snap?.bgn || {};
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
          .map(([kk, vv]) => `${kk} ${Array.isArray(vv) ? vv.map(x => fmt(x, 5)).join(",") : fmt(vv, 5)}`)
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
      if (p === 15 && det.p15) extra = `<div class="logic">details: ${JSON.stringify(det.p15).slice(0, 110)}…</div>`;
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
      $("#nw-dropped").textContent =
        `${st.filtered || 0} dropped for no crypto relevance` +
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
        // honest empty state: 0.00 only prints when items actually scored
        if ((h.n_items || 0) === 0)
          return `<div class="kv"><span>${tf.toUpperCase()}</span>` +
            `<b class="dim">— <span class="dim">no live items (synthetic never scores)</span></b></div>`;
        return `<div class="kv"><span>${tf.toUpperCase()} ` +
          `<span class="dim">½${Math.round(h.half_life_min || 0)}m</span></span>` +
          `<b style="color:${(h.impact || 0) >= 0 ? "var(--green)" : "var(--red)"}">` +
          `${fmt(h.impact, 3)} <span class="dim">niv ${fmt(h.niv, 3)} smd ${fmt(h.smd, 3)}</span></b></div>`;
      }).join("");
      const h1 = hz.tf1;
      if (h1 && h1.niv !== undefined) pushNiv(h1.niv);
    } catch (e) { /* status optional */ }
    $("#nw-scope").textContent = `${S.symbol} · ${d.source || ""} · ${d.count ?? (d.items || []).length} items`;
    const cats = d.categories || {};
    $("#nw-agg").innerHTML = TF.map(tf => {
      const imp = (d.impacts || {})[tf];
      return `<div class="kv"><span>${tf.toUpperCase()} impact</span><b>${fmt(imp, 4)}</b></div>`;
    }).join("") + `<div class="kv"><span>aggregate</span><b>${fmt(d.aggregate, 4)}</b></div>
      <div class="kv"><span>polarity</span><b>${fmt(d.polarity, 3)}</b></div>`;
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
      if ((h.n_items || 0) === 0)
        return `<div class="kv"><span>${tf.toUpperCase()}</span>` +
          `<b class="dim">— <span class="dim">no live items</span></b></div>`;
      return `<div class="kv"><span>${tf.toUpperCase()}</span>` +
        `<b style="color:${(h.impact || 0) >= 0 ? "var(--green)" : "var(--red)"}">` +
        `${fmt(h.impact, 3)} <span class="dim">niv ${fmt(h.niv, 3)} smd ${fmt(h.smd, 3)}</span></b></div>`;
    }).join("");
    if (hz.tf1 && hz.tf1.niv !== undefined) pushNiv(hz.tf1.niv);
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
  const mx = Math.max(0.05, ...vs.map(Math.abs));
  x.strokeStyle = "#4fd1c5"; x.lineWidth = 1.5; x.beginPath();
  NIV_HIST.forEach((p, i) => {
    const px = (i / (NIV_HIST.length - 1)) * w;
    const py = h / 2 - (p.v / mx) * (h / 2 - 4);
    i ? x.lineTo(px, py) : x.moveTo(px, py);
  });
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
      <div class="kv"><span>agreement (4×)</span><b>${fmt(br.agreement, 3)}</b></div>
      <div class="kv"><span>experience EQ</span><b>${fmt(br.experience, 3)}</b></div>
      <div class="kv"><span>dominant state</span><b>${br.dominant || "—"} (${fmt(br.p_dominant, 2)})</b></div>
      <div class="kv"><span>hormones D/O/S/T</span><b>${fmt(br.hormones_avg?.dopamine, 2)} / ${fmt(br.hormones_avg?.octopamine, 2)} / ${fmt(br.hormones_avg?.serotonin, 2)} / ${fmt(br.hormones_avg?.tyramine, 2)}</b></div>`;
    const wrap = $("#br-instances"); wrap.innerHTML = "";
    (br.instances || []).forEach(inst => {
      const card = document.createElement("div");
      card.className = "card";
      const h = inst.hormones || {};
      card.innerHTML = `<h4><span>${inst.id}</span><span>${inst.h_signal >= 0 ? "+" : ""}${fmt(inst.h_signal, 4)}</span></h4>
        <div class="kv-row">
          <div class="kv"><span>state</span><b>${inst.dominant || "—"}</b></div>
          <div class="kv"><span>experience</span><b>${fmt(inst.experience, 3)}</b></div>
        </div>
        <div class="kv-row" style="margin-top:8px">
          ${["dopamine", "serotonin", "octopamine", "tyramine"].map(k =>
            `<div class="kv"><span>${k.slice(0, 4)} ${fmt(h[k], 2)}</span><div class="bar"><i class="${(h[k] ?? 0) > 0.5 ? "pos" : "neg"}" style="width:${clamp((h[k] ?? 0) * 100, 0, 100)}%"></i></div></div>`).join("")}
        </div>
        <div class="logic">pattern ${fmt(inst.pattern, 3)} · P(dominant) ${fmt(inst.p_dominant, 2)}</div>`;
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
      $("#br-verify").innerHTML = (fw.verify || []).map(v =>
        `<div class="kv"><span>${esc(v.step)}${v.ms != null ? ` <span class="dim">${v.ms} ms</span>` : ""}</span>` +
        `<b class="${v.ok ? "acc-up" : "acc-dn"}" title="${esc(v.detail || "")}">${esc(String(v.detail || "").slice(0, 90))}</b></div>`
      ).join("") || `<div class="dim">no verification yet</div>`;
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
          const cellv = v2 => {
            const a2 = v2 > 0 ? (Math.log1p(v2) / Math.log1p(mmax)) * 0.7 : 0;
            return `<td style="background:rgba(0,212,240,${a2.toFixed(3)});text-align:right;font-family:var(--mono);font-size:10px" title="${v2}">${v2 > 0 ? (v2 < 0.01 ? v2.toExponential(1) : v2.toFixed(2)) : "·"}</td>`;
          };
          const ntc = Object.entries(an.nt_census || an.mbon_nt || {})
            .filter(([k]) => k && k !== "?").slice(0, 6);
          anEl.innerHTML = `
            <div class="kv-row">
              <div class="kv"><span>neurons (cells)</span><b>${(an.neurons || 0).toLocaleString()}</b></div>
              <div class="kv"><span>connections (≥5 syn)</span><b>${(an.connections || 0).toLocaleString()}</b></div>
              <div class="kv"><span>synapses (Σ)</span><b>${an.synapses != null ? an.synapses.toLocaleString() : "computing…"}</b></div>
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
          anEl.innerHTML = `<div class="dim pad">anatomy pass running — full neurons/cells/synapses census in a few seconds…</div>`;
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
            return `<td style="background:${bg};text-align:right;font-family:var(--mono);font-size:9.5px" title="${v2}">${v2 !== 0 ? (Math.abs(v2) < 0.01 ? v2.toExponential(1) : v2.toFixed(3)) : "·"}</td>`;
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
        <div class="kv"><span>push-pull odd-sum</span><b class="${pp.odd_ok ? "acc-up" : "acc-dn"}">${fmt(pp.odd_sum, 4)}</b></div>
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
    $("#ag-auditor").innerHTML = `
      <div class="kv-row">
        <div class="kv"><span>daemon</span><b class="${aud.available ? "acc-up" : "acc-dn"}">${aud.available ? "● online" : "○ offline"}</b></div>
        <div class="kv"><span>model</span><b>${aud.model || "qwen2.5-coder:3b"}</b></div>
        <div class="kv"><span>triggers (exc / low-acc)</span><b>${aud.triggers?.exceptions ?? 0} / ${aud.triggers?.accuracy ?? 0}</b></div>
        <div class="kv"><span>audits run</span><b>${aud.triggers?.audits ?? 0}</b></div>
      </div>
      ${aud.available ? "" : `<div class="hint">${aud.hint || "run: bash scripts/setup_local_llm.sh"}</div>`}
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
          <div class="dim" style="font-size:10px;margin-top:4px">score ${pk.score != null
            ? `${Math.round(pk.score * 100)}% over ${pk.samples} outcomes`
            : (pk.samples ? `${pk.samples} samples` : "no outcomes yet")}${pk.imported ? " · IMPORTED" : ""}</div>
          <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap">
            <button class="btn" data-dl="${pk.id}">download pack</button>
          </div>
          <div class="dim" style="font-size:9.5px;margin-top:6px"><a href="${pk.setup}" target="_blank" rel="noopener">get API key ↗</a></div>
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
    $$("#ag-packs [data-dl]").forEach(b => b.addEventListener("click", async () => {
      try {
        const r = await fetch("/api/agents/pack/" + b.dataset.dl);
        const d = await r.json();
        if (!d.ok) throw new Error("pack");
        const blob = new Blob([JSON.stringify(d.pack, null, 2)],
                              { type: "application/json" });
        const a2 = document.createElement("a");
        a2.href = URL.createObjectURL(blob);
        a2.download = `agent-pack-${b.dataset.dl}.json`;
        a2.click();
        URL.revokeObjectURL(a2.href);
        toast("pack downloaded — import with your key below");
      } catch (e) { toast("download failed", "err"); }
    }));

    const slots = a.slots || [];
    const titles = ["Gemini primary", "Backup 1 · Gemini", "Backup 2 · GPT-4o", "Backup 3 · Claude"];
    $("#ag-slots").innerHTML = slots.map((s, i) => `
      <div class="card">
        <h4><span>${titles[i] || s.status}</span><span class="slot-status ${s.status}">${s.status}</span></h4>
        <div class="kv-row">
          <div class="kv"><span>key</span><b>${s.key || "(none)"}</b></div>
          <div class="kv"><span>calls / h</span><b>${s.usage ?? 0} / ${s.limit ?? "—"}</b></div>
        </div>
        ${s.last?.insight ? `<div class="logic">“${esc(s.last.insight)}”</div>` : ""}
        <button class="btn test-slot" data-slot="${["gemini", "backup1", "backup2", "backup3"][i]}" style="margin-top:10px">test</button>
      </div>`).join("");
    const lo = a.local || {};
    $("#ag-local").innerHTML = `
      <div class="kv-row">
        <div class="kv"><span>model</span><b>${lo.label || "—"}</b></div>
        <div class="kv"><span>available</span><b>${lo.available ? "yes" : "no"}</b></div>
        <div class="kv"><span>RAM tier</span><b>${lo.ram_gb ?? "?"} GB</b></div>
        <div class="kv"><span>last inference</span><b>${lo.last_ms ? lo.last_ms + " ms" : "—"}</b></div>
      </div>
      ${lo.error ? `<div class="hint">${lo.error}</div>` : ""}
      <button class="btn test-slot" data-slot="local" style="margin-top:10px">test</button>`;
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
    if (sset) for (const [id, k] of Object.entries(map)) {
      const el = $("#" + id); if (el && sset[k]) el.value = sset[k];
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

/* ═══════════ HISTORY ═══════════ */
async function loadHistory() {
  try {
    const [days, lb] = await Promise.all([api("/api/history/days"), api("/api/history/leaderboard")]);
    const dd = days.days || [];
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
    $("#hs-leader").innerHTML = (lb.leaderboard || []).map(r => {
      const nn = r.n ?? 0;
      const enough = nn >= 8;
      return `<div class="kv"><span title="formula F${r.formula ?? ""}">${esc(r.name || r.symbol || r.tf || ("F" + r.formula))}</span>` +
        (enough
          ? `<b class="${(r.accuracy ?? 0) >= 0.55 ? "acc-up" : (r.accuracy ?? 0) <= 0.45 ? "acc-dn" : ""}">${fx(r.accuracy ?? r.score, 4)} <span class="dim" style="font-weight:400">n=${nn}</span></b>`
          : `<b class="dim">— n=${nn} (needs ≥8)</b>`) + `</div>`;
    }).join("")
      || `<div class="dim pad">no resolved predictions yet</div>`;
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
            : (pr.issued ? "open / no close" : "no trade"),
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

async function loadSettings() {
  try {
    loadRL().catch(() => {});
    loadTrading().catch(() => {});
    const l = await api("/api/learning");
    const lr = l.learning || {};
    $("#st-eta").textContent = fmt(lr.eta ?? lr.rates?.eta ?? 0.02, 3);
    $("#st-alert").textContent = lr.high_alert ? "ON" : "off";
    $("#st-shares").textContent = Object.entries(l.shares || {}).map(([k, v]) => `${k}:${fmt(v, 2)}`).join(" · ") || "—";
  } catch { /* optional */ }
}

/* ═══════════ boot overlay + timers ═══════════ */
let bootDeadline = 0;
function renderBoot(snap) {
  const ov = $("#boot-overlay");
  if (snap.phase !== "analysis") { ov.classList.add("gone"); return; }
  ov.classList.remove("gone");
  const an = snap.analysis || {};
  bootDeadline = Date.now() + (an.remaining_s ?? 0) * 1000;
  const rem = an.remaining_s ?? 0;
  $("#boot-clock").textContent = `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
  const ready = an.readiness || {};
  $$("#boot-checks li").forEach(li => {
    const k = li.dataset.k;
    const ok = ready[k] === true || k === "price" ||
      (k === "bgn" && ready.formulas) || (k === "aggregation" && ready.weights);
    li.classList.toggle("ok", !!ok);
  });
  $("#boot-warning").classList.toggle("hidden", !an.warning);
}
/* ═══════ THE one setInterval (advice H-2): boot clock + quad countdown +
   REST safety net. Multiple timers were the "glitchy countdown" bug. ═══ */
let _tickN = 0;
setInterval(() => {
  _tickN++;
  // realised-path ring for the probability-branches chart (1 Hz, capped)
  try {
    const _pp = S.prices?.[S.symbol];
    if (_pp > 0) {
      if (!S.priceHist) S.priceHist = [];
      S.priceHist.push({ t: Date.now() / 1000, p: _pp });
      if (S.priceHist.length > 900) S.priceHist = S.priceHist.slice(-900);
    }
  } catch (e2) { /* ring optional */ }
  // 1) boot countdown
  const ov = $("#boot-overlay");
  if (ov && !ov.classList.contains("gone") && bootDeadline) {
    const rem = Math.max(0, (bootDeadline - Date.now()) / 1000);
    const bc = $("#boot-clock");
    if (bc) bc.textContent = `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
  }
  // 2) REST safety net every other tick (~2s) — keeps header honest
  if (_tickN % 2 === 0 && Date.now() - S.lastMsgAt >= 4000) {
    api("/api/snapshot").then(applySnapshot).catch(() => {});
  }
  // 2b) candle series refresh (~30 s): the realised line's backfill
  // must stay current or the charts drift from the tape again
  if (_tickN % 30 === 0) loadCandles().catch(() => {});
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
}, 1000);

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
      const isNow = !!c.issued;
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
  try { checkIssuedAlerts(snap); } catch (e) { /* alerts optional */ }
  S.serverOffset = (snap.ts ? snap.ts * 1000 : Date.now()) - Date.now();
  renderBoot(snap);               // overlay FIRST — contract test guard
  renderAlert(snap);
  try { renderQuads(); renderWhy(); renderRail(); } catch (e) { console.error("render", e); }
  try { renderBranches(); } catch (e) { console.error("branches", e); }
  try { drawMainChart(); } catch (e) { console.error("main chart", e); }
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
    fchip.title = `source ${f.source} · last delivery ${f.last_delivery_age_s ?? "?"}s ago · ${f.updates} updates · ${f.switches} switches`;
  }
  // §10.3 status chips — every subsystem's honesty state, one glance
  const sc = $("#status-chips");
  if (sc && snap.chips) {
    const c = snap.chips;
    const chip = (label, cls, tip) =>
      `<span class="chip ${cls}" title="${esc(tip || "")}">${label}</span>`;
    sc.innerHTML =
      chip(c.feed === "live" ? "● LIVE FEED" : "⚠ SIM FEED",
           c.feed === "live" ? "chip-ok" : "chip-warn",
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
  }
  refreshActiveView();
}
function onTick(msg) {
  S.prices = msg.prices || S.prices;
  const p = msg.prices?.[S.symbol];
  if (p != null) TF.forEach(tf => S.charts[tf]?.tick(p));
  drawMainChart();                  // main chart always ends at live price
  renderQuads({ heavy: false });   // 1 Hz: text only, no table rebuilds
}
function handleMessage(msg) {
  S.lastMsgAt = Date.now();
  if (msg.type === "NEWS") onNewsTicker(msg);
  else if (msg.type === "news_emergency" || msg.type === "news_emergency_ack") {
    toast(`⚡ NEWS_EMERGENCY ${msg.asset || ""}: ${msg.reason || ""}`, "err");
    if (S.view === "news") loadNews();
  }
  else if (msg.type === "snapshot") applySnapshot(msg);
  else if (msg.type === "tick") onTick(msg);
  else if (msg.type === "phase") renderBoot({ phase: msg.phase, analysis: msg });
  else if (msg.type === "sl_event") toast(msg.message || msg.kind, msg.kind === "close" ? "err" : "warn");
  else if (msg.type === "analyse_stage") onAnalyseStage(msg);
  else if (msg.type === "analyse_done") onAnalyseDone(msg);
}
let ws;                       // module scope: master tick guards silence
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  try { ws = new WebSocket(`${proto}://${location.host}/ws`); } catch { return; }
  ws.onopen = () => {
    S.wsRetry = 0;
    S.wsLast = Date.now();
    try { ws.send(JSON.stringify({ type: "HELLO", view: S.view })); } catch {}
    // §10.4 silence is checked by the SINGLE master tick (H-2): a socket
    // silent for 15s is closed there — never wait the browser's 30-minute
    // timeout (error J-7)
  };
  ws.onmessage = () => { S.wsLast = Date.now(); };
  ws.onclose = () => {
    S.wsRetry++;
    setTimeout(connectWS, Math.min(8000, 1000 * S.wsRetry));
  };
  ws.onmessage = ev => { try { handleMessage(JSON.parse(ev.data)); } catch { /* partial */ } };
}

/* ═══════════ events ═══════════ */
function bindGlobal() {
  $("#nw-refresh")?.addEventListener("click", loadNews);
  $("#nw-drill")?.addEventListener("click", drillEmergency);
  $("#hs-reload")?.addEventListener("click", loadHistory);
  $("#ag-reload")?.addEventListener("click", loadAgents);
  $("#ag-save")?.addEventListener("click", async () => {
    const body = {
      gemini_key: $("#key-gemini").value.trim(), backup1_key: $("#key-backup1").value.trim(),
      backup2_key: $("#key-backup2").value.trim(), backup3_key: $("#key-backup3").value.trim(),
      newsapi_key: $("#key-newsapi").value.trim(), neuprint_key: $("#key-neuprint").value.trim(),
    };
    try {
      await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      toast("keys saved"); loadAgents();
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
let _errShown = 0, _fetchFails = 0;
function _netBanner(show, detail) {
  let b = $("#net-banner");
  if (!show) { if (b) b.remove(); return; }
  if (!b) {
    b = document.createElement("div");
    b.id = "net-banner";
    b.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:300;background:#2a0d14;" +
      "color:#f0385a;border-bottom:1px solid #f0385a;padding:8px 16px;font:600 12px monospace;" +
      "letter-spacing:1px;text-align:center";
    document.body.appendChild(b);
  }
  b.textContent = detail || "ENGINE OFFLINE — retrying every 2s (start it: bash run.sh)";
}
const _origFetch = window.fetch;
window.fetch = async (...args) => {
  try {
    const r = await _origFetch(...args);
    if (r.ok) { _fetchFails = 0; _netBanner(false); }
    else if (String(args[0]).includes("/api/")) {
      if (++_fetchFails >= 3) _netBanner(true, `ENGINE ERROR ${r.status} — check server log`);
    }
    return r;
  } catch (e) {
    if (String(args[0]).includes("/api/") && ++_fetchFails >= 3)
      _netBanner(true, "ENGINE OFFLINE — retrying (if in Codespaces: bash run.sh)");
    throw e;
  }
};
window.addEventListener("error", e => {
  if (_errShown++ < 3) toast("JS: " + (e.message || "error"), "err");
});
window.addEventListener("unhandledrejection", e => {
  if (_errShown++ < 3) toast("async: " + (e.reason?.message || e.reason || "rejected"), "err");
});

/* ═══════════ boot ═══════════ */
(function boot() {
  try { S.tfFocus = localStorage.getItem("tfFocus") === "1"; } catch (e) { S.tfFocus = false; }
  buildRail(); buildTabs(); buildToolbar(); buildQuads(); bindGlobal(); bindAnalyse(); bindAutoChip(); bindChartSrc(); bindTrading(); bindRiskCalc(); syncAnalyseButton(); bindAllTf(); renderDashboardMode();
  connectWS();
  api("/api/snapshot").then(applySnapshot).catch(() => {});
  // REST safety net lives in the single master setInterval below (H-2)
})();

// [binance-header-btn] discoverable exchange entry point in the header.
(() => {
  const b = $("#btn-binance"); if (!b) return;
  b.addEventListener("click", () => {
    document.querySelector('.tab[data-t="settings"]')?.click();
    setTimeout(() => $("#trade-enable")?.scrollIntoView({behavior:"smooth", block:"center"}), 80);
    toast("Binance: paste keys → Enable → ARM. Nothing trades until a qualifying prediction is issued.", "ok");
  });
  const paint = async () => {
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

