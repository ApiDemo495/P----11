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
const EMO = ["FOMO","Greed","Fear","Panic","Hope","Euphoria","Anxiety","Confusion",
  "Regret","Trust","Doubt","Excitement","Boredom","Guilt","Pride","Envy","Relief",
  "Anticipation","Denial","Euphoria 2","Greed 2","CERC"];
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
  21:"CCI master aggregate: 31 weighted components",
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
  symbol: "BTC", focusTf: "tf1", view: "dashboard",
  snap: null, prices: {}, lastMsgAt: 0, wsRetry: 0,
  cycles: {}, grcFlash: {}, lastSec: {},
  charts: {}, spark: {},
  newsSel: "BTC", histDay: null,
};

/* ═══════════ ChartX — own canvas line chart ═══════════ */
class ChartX {
  constructor(el, opts = {}) {
    this.el = el; this.spark = !!opts.spark;
    this.cv = document.createElement("canvas");
    this.cv.style.cssText = "width:100%;height:100%;display:block";
    el.innerHTML = ""; el.appendChild(this.cv);
    this.bars = []; this.lines = []; this.markers = [];
    if (typeof ResizeObserver !== "undefined")
      new ResizeObserver(() => this.draw()).observe(el);
  }
  setData(bars) { this.bars = bars.slice(-260); this.draw(); }
  setLines(l) { this.lines = l || []; this.draw(); }
  setMarkers(m) { this.markers = m || []; this.draw(); }
  tick(price) {
    const t = Math.floor(Date.now() / 1000 / 60) * 60;
    const b = this.bars[this.bars.length - 1];
    if (!b) this.bars.push({ time: t, open: price, high: price, low: price, close: price });
    else if (b.time === t) { b.close = price; b.high = Math.max(b.high, price); b.low = Math.min(b.low, price); }
    else if (b.time < t) this.bars.push({ time: t, open: b.close, high: Math.max(b.close, price), low: Math.min(b.close, price), close: price });
    else return;
    if (this.bars.length > 280) this.bars.shift();
    this.draw();
  }
  draw() {
    const w = this.el.clientWidth || 240, h = this.el.clientHeight || 100;
    const cv = this.cv; cv.width = w * 2; cv.height = h * 2;
    const g = cv.getContext("2d"); g.setTransform(2, 0, 0, 2, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!this.bars.length) return;
    let hi = -Infinity, lo = Infinity;
    for (const b of this.bars) { hi = Math.max(hi, b.high); lo = Math.min(lo, b.low); }
    if (!this.spark) for (const l of this.lines) { hi = Math.max(hi, l.price); lo = Math.min(lo, l.price); }
    const pad = (hi - lo) * 0.08 || Math.max(hi * 1e-4, 0.01); hi += pad; lo -= pad;
    const rw = this.spark ? 2 : 48;
    const X = i => (i / Math.max(1, this.bars.length - 1)) * (w - rw) + 1;
    const Y = p => h - 3 - ((p - lo) / (hi - lo || 1)) * (h - 6);
    if (this.spark) {
      g.beginPath();
      this.bars.forEach((b, i) => { const x = X(i), y = Y(b.close); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.strokeStyle = "rgba(0,212,240,.45)"; g.lineWidth = 1; g.stroke(); return;
    }
    for (const l of this.lines) {
      g.strokeStyle = l.color; g.setLineDash(l.dash === false ? [] : [5, 4]); g.lineWidth = l.hit ? 1.8 : 1;
      g.beginPath(); g.moveTo(0, Y(l.price)); g.lineTo(w - rw, Y(l.price)); g.stroke(); g.setLineDash([]);
      g.fillStyle = l.color; g.font = "9px monospace"; g.textAlign = "left";
      g.fillText(`${l.title} ${l.price.toFixed(l.price > 1000 ? 0 : 2)}`, 4, Y(l.price) - 3);
    }
    g.beginPath();
    this.bars.forEach((b, i) => { const x = X(i), y = Y(b.close); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.strokeStyle = "#00d4f0"; g.lineWidth = 1.3; g.stroke();
    g.lineTo(X(this.bars.length - 1), h - 3); g.lineTo(X(0), h - 3); g.closePath();
    g.fillStyle = "rgba(0,212,240,.07)"; g.fill();
    for (const m of this.markers) {
      const i = this.bars.findIndex(b => b.time === m.time); if (i < 0) continue;
      g.fillStyle = m.color; g.font = "9px sans-serif"; g.textAlign = "center";
      g.fillText(m.glyph || "•", X(i), m.above ? Y(this.bars[i].high) - 3 : Y(this.bars[i].low) + 10);
    }
    const last = this.bars[this.bars.length - 1], ly = Y(last.close);
    g.fillStyle = last.close >= last.open ? "#00d4f0" : "#f0385a";
    g.fillRect(w - rw + 1, ly - 7, rw - 2, 14);
    g.fillStyle = "#04040c"; g.font = "bold 9px monospace"; g.textAlign = "left";
    g.fillText(px(last.close), w - rw + 3, ly + 3);
    g.fillStyle = "#55556e"; g.font = "8px monospace";
    g.fillText(px(hi), w - rw + 2, 10); g.fillText(px(lo), w - rw + 2, h - 5);
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
  S.symbol = sym;
  $$(".asset-btn").forEach(b => b.classList.toggle("active", b.dataset.sym === sym));
  syncToolbar();
  refreshActiveView(true);
  buildQuads(); loadCandles();
}
function buildTabs() {
  $$(".tab-btn").forEach(b => b.addEventListener("click", () => {
    S.view = b.dataset.view;
    $$(".tab-btn").forEach(x => x.classList.toggle("active", x === b));
    $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + S.view));
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
  S.focusTf = tf; syncToolbar(); renderWhy(); renderQuads();
  refreshActiveView(true);
}

function refreshActiveView(force) {
  if (S.view === "dashboard") { renderWhy(); renderQuads(); }
  else if (S.view === "formulas") loadFormulas();
  else if (S.view === "emotion") loadEmotion();
  else if (S.view === "physics") loadPhysics();
  else if (S.view === "news") loadNews();
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
      <div class="q-spark" id="qs-${tf}"></div>
      <div class="q-price" id="qp-${tf}">—</div>
      <div class="q-dir flat" id="qd-${tf}">—</div>
      <div class="q-tps" id="qt-${tf}"></div>
      <div class="q-foot"><span id="qf-${tf}"></span><span id="qa-${tf}"></span></div>`;
    grid.appendChild(q);
    S.charts[tf] = new ChartX($("#qc-" + tf));
    S.spark[tf] = new ChartX($("#qs-" + tf), { spark: true });
  });
  renderQuads(); loadCandles();
}
function renderQuads() {
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
      tp.innerHTML = (cyc?.targets || []).map((t, i) =>
        `<span class="tp ${t.status === "hit" ? "hit" : t.status === "locked" ? "locked" : ""}">TP${i + 1}${t.status === "hit" ? " ✓" : t.status === "locked" ? " 🔒" : ""} ${t.level ? px(t.level) : ""}</span>`
      ).join("") + (cyc?.sl ? `<span class="tp sl">SL ${px(cyc.sl)}</span>` : "");
    }
    // price line
    const pr = S.prices[S.symbol] ?? a.price;
    const chg = a.window_change_pct ?? 0;
    $("#qp-" + tf).innerHTML = pr != null
      ? `${px(pr)} <span class="chg ${chg >= 0 ? "up" : "down"}">${chg >= 0 ? "+" : ""}${fmt(chg, 2)}%</span>` : "—";
    // TP/SL overlay lines on the chart
    const lines = [];
    if (cyc?.sl) lines.push({ price: cyc.sl, color: "#f0385a", title: "SL" });
    (cyc?.targets || []).forEach((t, i) => {
      if (t.level) lines.push({
        price: t.level,
        color: t.status === "hit" ? "#22c98a" : t.status === "locked" ? "#c9a227" : "#00d4f0",
        title: `TP${i + 1}${t.status === "hit" ? " ✓" : t.status === "locked" ? " 🔒" : ""}`,
        dash: t.status !== "locked", hit: t.status === "hit",
      });
    });
    S.charts[tf]?.setLines(lines);
  });
  renderRail();
  try { renderCalls(); } catch (e) { console.error("calls", e); }
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
  el.innerHTML = `
    <div class="why-head">
      <div class="why-dir ${dir > 0 ? "up" : dir < 0 ? "down" : "flat"}">
        ${dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO TRADE"}<span class="sym">${S.symbol} · ${tf.toUpperCase()} · ${TF_META[tf].mins}m</span></div>
      <div class="why-conf">CONFIDENCE
        <div class="bar"><i class="${s.confidence > 0.35 ? "pos" : "neg"}" style="width:${clamp((s.confidence || 0) * 100, 0, 100)}%"></i></div>
        <b>${fmt(s.confidence, 3)}</b> / floor ${bd.floor ?? 0.35}</div>
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
        <div class="kv"><span>1 · |CCI_adj| × min(1, DRHE ${fmt(m.drhe, 2)})</span><b>${fmt(bd.base_confidence, 4)}</b></div>
        <div class="kv"><span>2 · × (0.7 + 0.3 × EQ ${fmt(m.eq, 2)})</span><b>brain B2</b></div>
        <div class="kv"><span>3 · state "${m.dominant || "—"}" ${m.state_aligned ? "aligned → +" + fmt(0.1 * (m.p_dominant || 0), 3) : "not aligned"}</span><b>${m.state_aligned ? "aligned" : "—"}</b></div>
        <div class="kv"><span>4 · RTP ${fmt(m.rtp, 2)}${m.rtp > 0.6 ? " ⚠" : ""} · PCV ${fmt(m.pcv, 2)}${m.pcv < 0 ? " ⚠" : ""}</span><b>penalties</b></div>
        <div class="kv"><span>5-7 · ×(0.5+0.5·WFCP ${fmt(m.wfcp, 2)}) ×(1−DRPT ${fmt(m.drpt, 2)}) ×(1−GSPI ${fmt(m.gspi, 2)})</span><b>${fmt(bd.final_confidence, 3)}</b></div>
        <h5 style="margin-top:12px">SUBSYSTEM VOTES</h5>
        <div class="votes">${Object.entries(votes).map(([k, v]) =>
          `<span class="vote ${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${k} ${v > 0 ? "▲" : v < 0 ? "▼" : "·"}</span>`).join("")}</div>
        <div class="hint">4-vote double-check: PCV + agent + brain (+ emotion/physics here) must agree to issue.</div>
      </div>
    </div>`;
}

/* ═══════════ CALLS & TARGETS — TP/SL for every timeframe ═══════════ */
function renderCalls() {
  const el = $("#calls-panel"); if (!el) return;
  const a = S.snap?.assets?.[S.symbol] || {};
  $("#calls-scope").textContent = `${S.symbol} · every main timeframe (sub-windows are TF targets, no separate toggle)`;
  const rows = TF.map(tf => {
    const cyc = a.tfs?.[tf];
    const res = S.snap?.systems?.[`${S.symbol}|${tf}`];
    const m = TF_META[tf];
    const dir = cyc?.direction ?? res?.direction ?? 0;
    const dirTxt = dir > 0 ? '<b class="up">▲ BULL</b>' : dir < 0 ? '<b class="down">▼ BEAR</b>' : '<b class="flat">● FLAT</b>';
    const tps = (cyc?.targets || []).map((t, i) => {
      const st = t.status === "hit" ? '<i class="st hit">HIT</i>'
        : t.status === "locked" ? '<i class="st lock">LOCK</i>' : '<i class="st open">open</i>';
      return `<td class="tp-cell">${t.level ? px(t.level) : "—"}<br/><span class="tp-min">TP${i + 1} · ${m.tps[i]}m</span> ${st}</td>`;
    }).join("");
    const sl = cyc?.sl ? `${px(cyc.sl)}` : "—";
    return `<tr class="${tf === S.focusTf ? "focus-row" : ""}">
      <td class="mono">${tf.toUpperCase()}</td>
      <td class="mono">${m.mins}m</td>
      <td>${dirTxt} <span class="dim">${cyc?.issued ? "· issued" : "· no trade"}</span></td>
      <td class="mono">${fmt(res?.confidence, 3)}</td>
      ${tps}
      <td class="mono sl-cell">${sl}</td>
      <td class="mono">${cyc?.ratio ? "1:" + cyc.ratio : "—"}</td>
    </tr>`;
  }).join("");
  el.innerHTML = `<table class="tbl calls-tbl">
    <thead><tr><th>TF</th><th>WINDOW</th><th>CALL</th><th>CONF</th>
      <th>TP1 ${px(0)}</th><th>TP2</th><th>TP3</th><th>TP4</th><th>SL</th><th>RATIO</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="hint">Each main timeframe issues its own call with 4 sub-targets (the "sub timeframes" — TP minutes per window) and one stop-loss. Levels come from the engine's ATR projection at every GRC; status updates live.</div>`;
  // fix header TP labels with real first-target context (keep simple labels)
  const ths = $$(".calls-tbl th");
  if (ths.length >= 8) { ths[4].textContent = "TP1"; ths[5].textContent = "TP2"; ths[6].textContent = "TP3"; ths[7].textContent = "TP4"; }
}

/* ═══════════ candles (per symbol, per TF) ═══════════ */
async function loadCandles() {
  try {
    await Promise.all(TF.map(async tf => {
      const d = await api(`/api/candles?symbol=${S.symbol}&tf=${tf}`);
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
      c.setData(bars.map(x => ({ time: x.t, open: x.o, high: x.h, low: x.l, close: x.c })));
      c.setMarkers(markers.slice(-36));
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
      card.innerHTML = `<h4><span>F${fid} · ${name}</span><span>${contrib >= 0 ? "+" : ""}${contrib != null ? fmt(contrib, 4) : "—"}</span></h4>
        <div class="kv-row">
          <div class="kv"><span>raw</span><b>${raw >= 0 ? "+" : ""}${fmt(raw, 3)}</b></div>
          <div class="kv"><span>Bgn-adj</span><b>${adj >= 0 ? "+" : ""}${fmt(adj, 3)}</b></div>
          <div class="kv"><span>weight ω</span><b>${fmt(w, 4)}</b></div>
        </div>
        ${hist.length > 2 ? `<canvas class="spark" height="46" data-f="${fid}"></canvas>
        <div class="kv-row"><div class="kv"><span>min/mean/max (100 GRC)</span><b>${fmt(mn, 3)} / ${fmt(mean, 3)} / ${fmt(mx, 3)}</b></div></div>` : ""}
        <div class="logic">${FORMULA_LOGIC[fid] || ""}</div>`;
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
    drawSparks(fh.core || {});
  } catch (e) { toast("formulas: " + e.message, "err"); }
}
function drawSparks(core) {
  $$("#fl-cards canvas.spark").forEach(cv => {
    const fid = cv.dataset.f;
    const series = core[fid] || [];
    if (series.length < 3) return;
    const w = cv.clientWidth || 220; cv.width = w * 2;
    const g = cv.getContext("2d"); g.scale(2, 1);
    const mn = Math.min(...series), mx = Math.max(...series), rng = (mx - mn) || 1;
    g.strokeStyle = "#c9a227"; g.lineWidth = 1.3; g.beginPath();
    series.slice(-120).forEach((v, i, arr) => {
      const x = (i / (arr.length - 1)) * w, y = 42 - ((v - mn) / rng) * 38;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  });
}

/* ═══════════ EMOTION ═══════════ */
async function loadEmotion() {
  try {
    const sys = await api(`/api/systems?symbol=${S.symbol}`);
    const em = sys.systems?.[`${S.symbol}|${S.focusTf}`]?.emotion || {};
    $("#em-scope").textContent = `${S.symbol} · ${S.focusTf.toUpperCase()}`;
    $("#em-narrative").innerHTML = `
      <div class="why-reason">${em.narrative || "waiting for the first emotion pass…"}</div>
      <div class="kv-row">
        <div class="kv"><span>CERC composite (E22)</span><b>${fmt(em.cerc, 4)}</b></div>
        <div class="kv"><span>aggregate</span><b>${fmt(em.aggregate, 4)}</b></div>
        <div class="kv"><span>RTP trauma</span><b>${fmt(em.rtp, 3)}${(em.rtp || 0) > 0.6 ? " ⚠" : ""}</b></div>
        <div class="kv"><span>TEDR → P6</span><b>${fmt(em.tedr, 4)}</b></div>
      </div>
      <div class="hint">22 spec emotion formulas feed E22 CERC, which votes inside the CCI. Circles: conic ring = |value|, border = direction.</div>`;
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
      const ring = $(".em-ring", cell);
      ring.style.setProperty("--p", `${clamp(Math.abs(v) * 100, 0, 100)}%`);
      $("i", ring).textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
      const tag = $(".em-tag", cell);
      let label = "NEUTRAL";
      if (Math.abs(v) > 0.7) label = "EXTREME"; else if (Math.abs(v) > 0.45) label = "ELEVATED";
      tag.textContent = label; tag.className = "em-tag " + label;
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
    for (let p = 1; p <= 22; p++) {
      const v = p === 22 ? ph.ufp : vals[String(p)];
      if (v == null) continue;
      const card = document.createElement("div");
      card.className = "card";
      let extra = "";
      if (p === 1 && det.p1) extra = `<div class="logic">predicted ${px(det.p1.p_predicted)} · σ ${fmt(det.p1.sigma_q, 1)} · Q ${fmt(det.p1.confidence_q, 2)}</div>`;
      if (p === 15 && det.p15) extra = `<div class="logic">details: ${JSON.stringify(det.p15).slice(0, 110)}…</div>`;
      card.innerHTML = `<h4><span>P${p} · ${P_NAMES[p] || ""}</span><span>${v >= 0 ? "+" : ""}${fmt(v, 4)}</span></h4>
        <div class="kv-row"><div class="kv"><span>value</span><b style="color:${v >= 0 ? "var(--green)" : "var(--red)"}">${v >= 0 ? "▲" : "▼"} ${fmt(v, 4)}</b></div></div>
        ${extra}<div class="logic">${P_LOGIC[p] || ""}</div>`;
      grid.appendChild(card);
    }
  } catch (e) { toast("physics: " + e.message, "err"); }
}

/* ═══════════ NEWS ═══════════ */
async function loadNews() {
  try {
    const d = await api(`/api/news/${S.symbol}`);
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
        <td style="font-family:inherit">${it.headline}</td>
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

/* ═══════════ BRAINS ═══════════ */
async function loadBrains() {
  try {
    const [geo, sys] = await Promise.all([
      api("/api/brain-geometry"),
      api(`/api/systems?symbol=${S.symbol}`),
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
            `<div class="kv"><span>${k.slice(0, 4)}</span><div class="bar"><i class="${(h[k] ?? 0) > 0.5 ? "pos" : "neg"}" style="width:${clamp((h[k] ?? 0) * 100, 0, 100)}%"></i></div></div>`).join("")}
        </div>
        <div class="logic">pattern ${fmt(inst.pattern, 3)} · P(dominant) ${fmt(inst.p_dominant, 2)}</div>`;
      wrap.appendChild(card);
    });
    renderCorel(s.corelation);
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
      ? `<b>DROSOPHILA MODE</b> — no keys needed. The 4-instance brain, 31 formulas, math & physics run everything. Paste any key below for an optional cloud second opinion.`
      : mode === "local" ? `<b>LOCAL MODEL</b> — on-device inference. Cloud keys optional.`
      : `<b>CLOUD AGENTS</b> — ${ (a.slots || []).filter(s => s.status === "connected").length } slot(s) live. Drosophila core still runs every GRC.`;
    const slots = a.slots || [];
    const titles = ["Gemini primary", "Backup 1 · Gemini", "Backup 2 · GPT-4o", "Backup 3 · Claude"];
    $("#ag-slots").innerHTML = slots.map((s, i) => `
      <div class="card">
        <h4><span>${titles[i] || s.status}</span><span class="slot-status ${s.status}">${s.status}</span></h4>
        <div class="kv-row">
          <div class="kv"><span>key</span><b>${s.key || "(none)"}</b></div>
          <div class="kv"><span>calls / h</span><b>${s.usage ?? 0} / ${s.limit ?? "—"}</b></div>
        </div>
        ${s.last?.insight ? `<div class="logic">“${s.last.insight}”</div>` : ""}
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
        ${sig.insight ? `<div class="logic">“${sig.insight}”</div>` : ""}`
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
    $("#hs-leader").innerHTML = (lb.leaderboard || []).map(r => `
      <div class="kv"><span>${r.symbol || r.tf}</span><b>${fmt(r.accuracy ?? r.score, 3)}</b></div>`).join("")
      || `<div class="dim pad">no resolved predictions yet</div>`;
    if (dd.length && !S.histDay) showDay(dd[0].date);
    else if (S.histDay) showDay(S.histDay);
  } catch (e) { toast("history: " + e.message, "err"); }
}
async function showDay(date) {
  try {
    S.histDay = date;
    $("#hs-day-label").textContent = date;
    const d = await api(`/api/history/day?date=${date}`);
    const rows = d.rows || d.predictions || [];
    $("#hs-detail").innerHTML = rows.length ? `
      <div class="day-row head"><span>TF</span><span>DIR</span><span>CONF</span><span>RESULT</span><span>TIME</span></div>` +
      rows.slice(-40).reverse().map(r => `
        <div class="day-row">
          <span>${(r.tf || "").toUpperCase()}</span>
          <span style="color:${r.direction > 0 ? "var(--green)" : r.direction < 0 ? "var(--red)" : "var(--dim)"}">${r.direction > 0 ? "▲" : r.direction < 0 ? "▼" : "●"}</span>
          <span>${fmt(r.confidence, 3)}</span>
          <span>${r.result || r.outcome || (r.issued ? "issued" : "no trade")}</span>
          <span class="dim">${r.time ? new Date(r.time * 1000).toLocaleTimeString() : ""}</span>
        </div>`).join("")
      : `<div class="dim pad">no prediction rows for this day</div>`;
  } catch (e) { toast("day: " + e.message, "err"); }
}

/* ═══════════ SETTINGS ═══════════ */
async function loadSettings() {
  try {
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
setInterval(() => {
  const ov = $("#boot-overlay");
  if (ov.classList.contains("gone") || !bootDeadline) return;
  const rem = Math.max(0, (bootDeadline - Date.now()) / 1000);
  $("#boot-clock").textContent = `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
}, 1000);

/* quadrant countdown rings — tick every second, independent of GRC */
setInterval(() => {
  const now = Date.now() / 1000;
  const a = S.snap?.assets?.[S.symbol] || {};
  TF.forEach(tf => {
    const cyc = a.tfs?.[tf]; if (!cyc) return;
    const q = $("#q-" + tf); if (!q) return;
    const total = Math.max(1, (cyc.cycle_end || 0) - (cyc.cycle_start || 0));
    const rem = clamp((cyc.cycle_end || 0) - now, 0, total);
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
}, 1000);

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
function applySnapshot(snap) {
  if (!snap || snap.type !== "snapshot") return;
  S.snap = snap;
  renderBoot(snap);
  renderAlert(snap);
  try { renderQuads(); renderWhy(); renderRail(); } catch (e) { console.error("render", e); }
  refreshActiveView();
}
function onTick(msg) {
  S.prices = msg.prices || S.prices;
  const p = msg.prices?.[S.symbol];
  if (p != null) TF.forEach(tf => S.charts[tf]?.tick(p));
  renderQuads();
}
function handleMessage(msg) {
  S.lastMsgAt = Date.now();
  if (msg.type === "snapshot") applySnapshot(msg);
  else if (msg.type === "tick") onTick(msg);
  else if (msg.type === "phase") renderBoot({ phase: msg.phase, analysis: msg });
  else if (msg.type === "sl_event") toast(msg.message || msg.kind, msg.kind === "close" ? "err" : "warn");
}
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws;
  try { ws = new WebSocket(`${proto}://${location.host}/ws`); } catch { return; }
  ws.onopen = () => { S.wsRetry = 0; };
  ws.onclose = () => { S.wsRetry++; setTimeout(connectWS, Math.min(8000, 1000 * S.wsRetry)); };
  ws.onmessage = ev => { try { handleMessage(JSON.parse(ev.data)); } catch { /* partial */ } };
}

/* ═══════════ events ═══════════ */
function bindGlobal() {
  $("#nw-refresh")?.addEventListener("click", loadNews);
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
  buildRail(); buildTabs(); buildToolbar(); buildQuads(); bindGlobal();
  connectWS();
  api("/api/snapshot").then(applySnapshot).catch(() => {});
  // REST fallback: keeps everything alive when WS is blocked by a proxy
  setInterval(async () => {
    if (Date.now() - S.lastMsgAt < 4000) return;
    try { applySnapshot(await api("/api/snapshot")); } catch { /* retry */ }
  }, 2000);
})();
