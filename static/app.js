/* ══════════════════════════════════════════════════════════════════════
   Section 12 — full dashboard application (9 tabs, sidebar layout)
   ══════════════════════════════════════════════════════════════════════ */
"use strict";

/* ─────────────────────────── helpers ─────────────────────────── */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const TF = ["tf1", "tf2", "tf3", "tf4"];
const TF_LABEL = { tf1: "TF1 · 22m", tf2: "TF2 · 44m", tf3: "TF3 · 130m", tf4: "TF4 · 247m" };
const ASSETS = ["BTC", "PAXG", "ETH", "BNB"];
const fmt = (v, d = 2) => (v == null || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
function toast(msg, kind = "ok") {
  const t = document.createElement("div");
  t.className = `toast ${kind === "warn" ? "warn" : kind === "err" ? "err" : ""}`;
  t.textContent = msg;
  $("#toast-layer").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}

/* ── Section 12 haptic patterns ── */
let HAPTICS = true, SOUND = false;
function haptic(kind) {
  if (!HAPTICS || !navigator.vibrate) return;
  const P = { lock: [50], tp: [40, 40, 60], sl: [200], close: [120, 60, 120, 60, 180], grc: [20], last10: [15] };
  navigator.vibrate(P[kind] || [20]);
}
function beep(freq = 880, ms = 60) {
  if (!SOUND) return;
  try {
    const ctx = beep._c || (beep._c = new AudioContext());
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = freq; o.connect(g); g.connect(ctx.destination);
    g.gain.value = 0.04; o.start(); o.stop(ctx.currentTime + ms / 1000);
  } catch { /* audio unavailable */ }
}

/* ─────────────────────────── tab navigation ─────────────────────────── */
const TAB_LOADERS = {};   // per-tab lazy loaders (filled below)
let ACTIVE_TAB = "dashboard";
$$(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".nav-btn").forEach(b => b.classList.toggle("active", b === btn));
    $$(".tab").forEach(t => t.classList.toggle("active", t.id === `tab-${btn.dataset.tab}`));
    ACTIVE_TAB = btn.dataset.tab;
    TAB_LOADERS[ACTIVE_TAB]?.();
  });
});

/* ═══════════════════════════ DASHBOARD ═══════════════════════════ */
const dash = {
  symbol: "BTC",
  focusTf: "tf1",
  charts: {}, qcharts: {}, priceLines: {}, qseries: {},
  lastGrc: 0, grcFlashAt: {},
  cycles: {}, alertState: "",
};

/* ── ChartX: our own canvas chart (TradingView lib removed — too slow).
   Live-line chart fed straight from the tick stream, with TP/SL lines,
   fractal/hit markers and a last-price tag. ───────────────────────── */
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
  setLines(lines) { this.lines = lines || []; this.draw(); }
  setMarkers(m) { this.markers = m || []; this.draw(); }
  updateBar(b) {
    const last = this.bars[this.bars.length - 1];
    if (last && last.time === b.time) this.bars[this.bars.length - 1] = b;
    else { this.bars.push(b); if (this.bars.length > 280) this.bars.shift(); }
    this.draw();
  }
  tick(price) {
    const n = Date.now() / 1000;
    const t = Math.floor(n / 60) * 60;
    const b = this.bars[this.bars.length - 1];
    if (!b) { this.bars.push({ time: t, open: price, high: price, low: price, close: price }); }
    else if (b.time === t) { b.close = price; b.high = Math.max(b.high, price); b.low = Math.min(b.low, price); }
    else if (b.time < t) this.bars.push({ time: t, open: b.close, high: Math.max(b.close, price),
                                          low: Math.min(b.close, price), close: price });
    else return;
    if (this.bars.length > 280) this.bars.shift();
    this.draw();
  }
  draw() {
    const w = this.el.clientWidth || 260, h = this.el.clientHeight || 110;
    const cv = this.cv; cv.width = w * 2; cv.height = h * 2;
    const g = cv.getContext("2d"); g.setTransform(2, 0, 0, 2, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!this.bars.length) return;
    let hi = -Infinity, lo = Infinity;
    for (const b of this.bars) { hi = Math.max(hi, b.high); lo = Math.min(lo, b.low); }
    if (this.spark) { const p = (hi - lo) * 0.1 || 1; hi += p; lo -= p; }
    else for (const l of this.lines) { hi = Math.max(hi, l.price); lo = Math.min(lo, l.price); }
    const pad = (hi - lo) * 0.06 || 1; hi += pad; lo -= pad;
    const rw = this.spark ? 2 : 46;                      // right price-tag gutter
    const X = i => (i / Math.max(1, this.bars.length - 1)) * (w - rw) + 2;
    const Y = p => h - 4 - ((p - lo) / (hi - lo)) * (h - 8);
    if (this.spark) {
      g.beginPath();
      this.bars.forEach((b, i) => { const x = X(i), y = Y(b.close); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.strokeStyle = "rgba(0,229,255,0.5)"; g.lineWidth = 1; g.stroke();
      return;
    }
    // TP / SL lines
    for (const l of this.lines) {
      g.strokeStyle = l.color; g.setLineDash(l.dash === false ? [] : [5, 4]);
      g.lineWidth = l.hit ? 2 : 1;
      g.beginPath(); g.moveTo(0, Y(l.price)); g.lineTo(w - rw, Y(l.price)); g.stroke();
      g.setLineDash([]);
      g.fillStyle = l.color; g.font = "9px monospace"; g.textAlign = "left";
      g.fillText(`${l.title} ${l.price.toFixed(1)}`, 4, Y(l.price) - 3);
    }
    // price line + soft area
    g.beginPath();
    this.bars.forEach((b, i) => { const x = X(i), y = Y(b.close); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.strokeStyle = "#00e5ff"; g.lineWidth = 1.3; g.stroke();
    g.lineTo(X(this.bars.length - 1), h - 4); g.lineTo(X(0), h - 4); g.closePath();
    g.fillStyle = "rgba(0,229,255,0.07)"; g.fill();
    // markers (fractals ◆, TP hits ▲)
    for (const m of this.markers) {
      let i = this.bars.findIndex(b => b.time === m.time);
      if (i < 0) continue;
      g.fillStyle = m.color; g.font = "9px sans-serif"; g.textAlign = "center";
      g.fillText(m.glyph || "•", X(i), m.above ? Y(this.bars[i].high) - 3 : Y(this.bars[i].low) + 10);
    }
    // last-price tag + hi/lo
    const last = this.bars[this.bars.length - 1], ly = Y(last.close);
    g.fillStyle = last.close >= last.open ? "#00e5ff" : "#ff1744";
    g.fillRect(w - rw + 1, ly - 7, rw - 2, 14);
    g.fillStyle = "#04040c"; g.font = "bold 9px monospace"; g.textAlign = "left";
    g.fillText(last.close >= 1000 ? last.close.toFixed(0) : last.close.toFixed(2), w - rw + 3, ly + 3);
    g.fillStyle = "#5a5a78"; g.font = "8px monospace";
    g.fillText(hi.toFixed(hi >= 1000 ? 0 : 2), w - rw + 2, 10);
    g.fillText(lo.toFixed(lo >= 1000 ? 0 : 2), w - rw + 2, h - 6);
  }
}

function buildCharts() {
  TF.forEach(tf => {
    const el = $(`#chart-${tf}`);
    if (el && !dash.charts[tf]) dash.charts[tf] = new ChartX(el);
    const qel = $(`#quantum-${tf}`);
    if (qel && !dash.qcharts[tf]) dash.qcharts[tf] = new ChartX(qel, { spark: true });
  });
}

async function loadCandles() {
  try {
    await Promise.all(TF.map(async tf => {
      const d = await api(`/api/candles?symbol=${dash.symbol}&tf=${tf}`);
      const bars = d.bars || [];
      const c = dash.charts[tf];
      if (!c || !bars.length) return;
      // fractal diamonds — 5-bar swing pivots (Section 12)
      const markers = [];
      for (let i = 2; i < bars.length - 2; i++) {
        const b = bars[i];
        if (b.h > bars[i - 1].h && b.h > bars[i - 2].h &&
            b.h > bars[i + 1].h && b.h > bars[i + 2].h)
          markers.push({ time: b.t, above: true, color: "#ff1744", glyph: "◆" });
        if (b.l < bars[i - 1].l && b.l < bars[i - 2].l &&
            b.l < bars[i + 1].l && b.l < bars[i + 2].l)
          markers.push({ time: b.t, above: false, color: "#00e5ff", glyph: "◆" });
      }
      // TP-hit markers from locked targets (feeds straight back from engine)
      const cyc = dash.cycles[`${dash.symbol}|${tf}`];
      (cyc?.targets || []).forEach(t => {
        if (t.status === "hit" && t.hit_ts)
          markers.push({ time: Math.floor(t.hit_ts / 60) * 60, above: true,
                         color: "#2ee6a8", glyph: "▲" });
      });
      c.setData(bars.map(x => ({ time: x.t, open: x.o, high: x.h, low: x.l, close: x.c })));
      c.setMarkers(markers.slice(-40));
      const lb = bars[bars.length - 1];
      if (lb) c.lastBar = { time: lb.t, open: lb.o, high: lb.h, low: lb.l, close: lb.c };
      dash.qcharts[tf]?.setData(
        bars.slice(-80).map(x => ({ time: x.t, open: 0, high: x.v || 0, low: 0, close: x.v || 0 })));
    }));
  } catch { /* candles endpoint may 404 before first GRC */ }
}

/* TP/SL overlay lines for every quadrant chart (from the live cycles) */
function applyTfLines() {
  TF.forEach(tf => {
    const c = dash.charts[tf]; if (!c) return;
    const cyc = dash.cycles[`${dash.symbol}|${tf}`];
    const lines = [];
    if (cyc?.sl) lines.push({ price: cyc.sl, color: "#ff1744", title: "SL" });
    (cyc?.targets || []).forEach((t, i) => {
      if (t.level) lines.push({
        price: t.level,
        color: t.status === "hit" ? "#2ee6a8" : t.status === "locked" ? "#c9a227" : "#00e5ff",
        title: `TP${i + 1}${t.status === "hit" ? " ✓" : t.status === "locked" ? " 🔒" : ""}`,
        dash: t.status !== "locked",
        hit: t.status === "hit",
      });
    });
    c.setLines(lines);
  });
}

function drawGauge(tf, conf) {
  const cv = $(`#gauge-${tf}`);
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, cx = W / 2, cy = H / 2, r = W / 2 - 12;
  const target = clamp(conf || 0, 0, 1);
  const cur = Number(cv.dataset.v || 0);
  const t0 = performance.now();
  cancelAnimationFrame(cv._raf || 0);
  const step = now => {
    const k = clamp((now - t0) / 500, 0, 1);
    const v = cur + (target - cur) * k;
    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 10; ctx.lineCap = "round";
    ctx.strokeStyle = "#1c1c2c";
    ctx.beginPath(); ctx.arc(cx, cy, r, 0.75 * Math.PI, 2.25 * Math.PI); ctx.stroke();
    // red → yellow → green gradient arc
    const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
    grad.addColorStop(0, "#ff1744"); grad.addColorStop(0.5, "#ffd600"); grad.addColorStop(1, "#2ee6a8");
    ctx.strokeStyle = grad;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0.75 * Math.PI, (0.75 + 1.5 * v) * Math.PI); ctx.stroke();
    ctx.fillStyle = "#e8e8f0"; ctx.font = "600 22px JetBrains Mono, monospace";
    ctx.textAlign = "center"; ctx.fillText(v.toFixed(2), cx, cy + 4);
    ctx.fillStyle = "#8a8aa8"; ctx.font = "9px Inter, sans-serif";
    ctx.fillText("CONFIDENCE", cx, cy + 20);
    if (k < 1) cv._raf = requestAnimationFrame(step); else cv.dataset.v = v;
  };
  cv._raf = requestAnimationFrame(step);
}

function renderQuadrant(tf, cyc, res) {
  if (!cyc) return;
  const qEl = $(`#q-${tf}`);
  // countdown ring + time
  const now = Date.now() / 1000;
  const total = Math.max(1, (cyc.cycle_end || 0) - (cyc.cycle_start || 0));
  const rem = clamp((cyc.cycle_end || 0) - now, 0, total);
  const frac = rem / total;
  const fg = $(".ring-fg", qEl);
  if (fg) fg.style.strokeDashoffset = (97.4 * (1 - frac)).toFixed(1);
  const mm = Math.floor(rem / 60), ss = Math.floor(rem % 60);
  $(".q-time", qEl).textContent = `${mm}:${String(ss).padStart(2, "0")}`;
  // Section 12: last-10-seconds haptic pulses
  if (rem <= 10 && rem > 0 && dash.lastSec?.[tf] !== Math.ceil(rem)) {
    dash.lastSec = dash.lastSec || {};
    dash.lastSec[tf] = Math.ceil(rem);
    haptic("last10");
  }
  // GRC cyan flash on reset
  if (cyc.cycle_start && dash.grcFlashAt[tf] !== cyc.cycle_start) {
    if (dash.grcFlashAt[tf]) { qEl.classList.add("cyan-flash"); setTimeout(() => qEl.classList.remove("cyan-flash"), 550); haptic("grc"); }
    dash.grcFlashAt[tf] = cyc.cycle_start;
  }
  // direction
  const dir = cyc.direction || 0;
  const dirEl = $(`#dir-${tf}`);
  const glyph = dir > 0 ? "▲ BULLISH" : dir < 0 ? "▼ BEARISH" : "● NO TRADE";
  const cls = dir > 0 ? "up" : dir < 0 ? "down" : "";
  if (dirEl.textContent.trim() !== glyph) {
    dirEl.innerHTML = `<span class="arrow slide">${glyph}</span>`;
    dirEl.className = `q-dir ${cls}`;
  }
  dirEl.style.opacity = cyc.issued ? 1 : 0.45;
  // ratio + TP list
  $("#ratio-" + tf).textContent = cyc.ratio ? `1 : ${cyc.ratio}` : "—";
  const tl = $(`#tplist-${tf}`);
  tl.innerHTML = "";
  (cyc.targets || []).forEach((t, i) => {
    const dot = document.createElement("span");
    const locked = t.status === "locked";
    dot.className = `tp-dot ${locked ? "hit" : ""} ${i === 3 ? "t4" : ""}`;   // Tp4 always filled
    dot.innerHTML = `<i></i>Tp${i + 1}${locked && t.level ? " " + fmt(t.level, 1) : ""}`;
    tl.appendChild(dot);
  });
  if (cyc.sl) {
    const sl = document.createElement("span");
    sl.className = "tp-dot"; sl.style.color = "var(--red)";
    sl.innerHTML = `<i style="border-color:var(--red)"></i>SL ${fmt(cyc.sl, 1)}`;
    tl.appendChild(sl);
  }
  // CCI + agent footers + live price
  const aset = dash.latest?.assets?.[dash.symbol];
  let pe = $("#q-price-" + tf, qEl);
  if (!pe) {
    pe = document.createElement("span");
    pe.id = "q-price-" + tf;
    const foot = qEl.querySelector(".q-foot");
    if (foot) foot.appendChild(pe);
  }
  pe.innerHTML = `<b style="color:var(--gold)">${fmt(aset?.price, 2)}</b>` +
    (aset ? ` <span style="color:${aset.window_change_pct >= 0 ? "var(--cyan)" : "var(--red)"}">` +
      `${aset.window_change_pct >= 0 ? "+" : ""}${aset.window_change_pct}%</span>` : "");
  if (res) {
    $("#q-cci-" + tf).textContent = `CCI ${fmt(res.cci_adjusted, 4)} · conf ${fmt(res.confidence, 3)}`;
    const ag = res.agent;
    $("#q-agent-" + tf).textContent = ag && ag.source !== "none"
      ? `AGENT ${ag.direction > 0 ? "+" : ag.direction < 0 ? "−" : "0"} (${ag.source}) ${fmt(ag.confidence, 2)}`
      : "AGENT —";
    drawGauge(tf, res.confidence);
    // dashed TP/SL lines on our canvas chart (solid when locked, ✓ when hit)
    applyTfLines();
  }
}

function updateAlertBar() {
  const bar = $("#alert-bar");
  let state = "";
  for (const s of ASSETS) {
    for (const tf of TF) {
      const c = dash.cycles[`${s}|${tf}`];
      if (!c) continue;
      if (c.close_alert) state = "close";
      else if (c.safe_alert && !state) state = "safe";
    }
  }
  if (state === dash.alertState) return;
  dash.alertState = state;
  bar.classList.remove("show-safe", "show-close", "hidden");
  if (!state) { bar.classList.add("hidden"); return; }
  if (state === "close") {
    bar.classList.add("show-close");
    $("#alert-icon").textContent = "✕"; $("#alert-text").textContent = "CLOSE POSITION IMMEDIATELY";
    haptic("close"); beep(440, 200);
  } else {
    bar.classList.add("show-safe");
    $("#alert-icon").textContent = "▲"; $("#alert-text").textContent = "SAFE — SL TIGHTENED 15%";
    haptic("sl"); beep(660, 120);
  }
}

function renderDashboard(snap) {
  dash.latest = snap;
  for (const s of ASSETS) {
    const a = snap.assets?.[s];
    if (!a) continue;
    for (const tf of TF) {
      const cyc = a.tfs?.[tf];
      if (!cyc) continue;
      const key = `${s}|${tf}`;
      dash.cycles[key] = cyc;
      if (s === dash.symbol) {
        const res = snap.systems?.[`${s}|${tf}`];
        renderQuadrant(tf, cyc, res);
      }
    }
  }
  $("#side-grc").textContent = snap.grc?.id ?? "—";
  const s2n = snap.grc?.seconds_to_next;
  $("#side-next").textContent = s2n != null ? `${Math.floor(s2n / 60)}m ${Math.floor(s2n % 60)}s` : "—";
  // learning LED
  $("#led-learn").className = "led " + (snap.learning?.high_alert ? "warn" : "on");
  $("#led-agent").className = "led " + (snap.agents?.slots?.[0]?.status === "connected" ? "on" : "off");
  updateModeChip(snap.agents?.mode);
  updateAlertBar();
  try { renderWhy(snap); } catch (e) { console.error("why", e); }
  // alert-bar cycle flags consumed after render
  Object.values(dash.cycles).forEach(c => { c.safe_alert = false; c.close_alert = false; });
}

/* asset buttons per quadrant */
TF.forEach(tf => {
  const holder = $(`.q-assets[data-tf="${tf}"]`);
  if (!holder) return;
  ASSETS.forEach(s => {
    const b = document.createElement("button");
    b.className = "asset-btn" + (s === dash.symbol ? " active" : "");
    b.textContent = s;
    b.addEventListener("click", () => {
      dash.symbol = s;
      $$(".asset-btn").forEach(x => x.classList.toggle("active", x.textContent === s));
      loadCandles();
      TAB_LOADERS.formulas?.(); TAB_LOADERS.emotion?.(); TAB_LOADERS.physics?.();
      $("#news-symbol").value = s; TAB_LOADERS.news?.();
    });
    holder.appendChild(b);
  });
});

/* ═══════════════════════════ ANALYSIS OVERLAY ═══════════════════════════ */
function updateOverlay(msg, snap) {
  const ov = $("#analysis-overlay");
  const phase = snap?.phase ?? msg?.phase;
  if (phase === "live") { ov.classList.add("gone"); return; }
  ov.classList.remove("gone");
  const rem = snap?.analysis?.remaining_s ?? msg?.remaining_s ?? 0;
  window.__ovDeadline = Date.now() + rem * 1000;   // local ticker between updates
  $("#analysis-clock").textContent = `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
  const ready = snap?.analysis?.readiness ?? msg?.readiness ?? {};
  $$("#analysis-checks li").forEach(li => {
    const k = li.dataset.k;
    const ok = ready[k] === true || (k === "price") || (k === "bgn" && ready.formulas) || (k === "aggregation" && ready.weights);
    li.classList.toggle("ok", !!ok);
  });
  const warn = snap?.analysis?.warning ?? msg?.warning;
  $("#analysis-warning").classList.toggle("hidden", !warn);
}

/* ═══════════════════════════ BRAIN MONITOR ═══════════════════════════ */
const brainsTab = { mode3d: true, loaded: false, actAt: 0 };
async function loadBrains3d() {
  if (brainsTab.loaded) return true;
  try { await import("/brains3d.js"); brainsTab.loaded = true; } catch { brainsTab.loaded = false; }
  return brainsTab.loaded;
}
async function refreshActivations(force) {
  const now = Date.now();
  if (!force && now - brainsTab.actAt < 8000) return;
  brainsTab.actAt = now;
  try {
    const j = await api(`/api/brain-activations?symbol=${dash.symbol}&tf=tf1`);
    if (window.brain3D?.ready) window.brain3D.setActivations(j.per_instance);
  } catch { /* retry next GRC */ }
}
TAB_LOADERS.brains = async () => {
  try {
    const g = await api("/api/brain-geometry");
    $("#brain-geom").textContent = `units ${g.units ?? "7700"} · edges ${g.edges ?? "—"}`;
    if (brainsTab.mode3d && await loadBrains3d()) {
      $("#brains2d").style.display = "none";
      $("#brains-canvas").style.display = "block";
      window.brain3D?.setVisible(true);
      refreshActivations(true);
      brainsTab.ready = true;
    } else {
      window.brain3D?.setVisible(false);
      $("#brains-canvas").style.display = "none";
      $("#brains2d").style.display = "grid";
      renderBrains2d();
    }
  } catch (e) { toast("brain geometry: " + e.message, "err"); }
  if (dash.latest) { renderCorel(dash.latest); renderWiring(dash.latest); }
};
$("#brain-flip")?.addEventListener("click", () => {
  brainsTab.mode3d = !brainsTab.mode3d;
  brainsTab.ready = false;
  TAB_LOADERS.brains();
});
$("#brain-refresh")?.addEventListener("click", () => TAB_LOADERS.brains());

function renderBrains2d() {
  const box = $("#brains2d");
  box.innerHTML = TF.map(tf => `
    <div class="brain2d-card">
      <b>${TF_LABEL[tf]}</b><br/>4 instances × 7,700 units<br/>
      <span style="color:var(--dim)">live activations via snapshot</span>
    </div>`).join("");
}

function updateBrains(snap) {
  const st = snap.brains || {};
  const cards = $("#brain-cards");
  if (!cards.dataset.built) {
    cards.dataset.built = "1";
    cards.innerHTML = [0, 1, 2, 3].map(i => `
      <div class="brain-card" id="brain-card-${i}">
        <h4>DB-${i + 1}</h4><div class="role">—</div>
        <div class="sig">—</div>
        <div class="horm-bar"><i style="width:50%"></i></div>
        <div class="dom-chip">—</div>
      </div>`).join("");
  }
  const sys = snap.systems?.[`${dash.symbol}|tf1`];
  const insts = sys?.brain?.instances || [];
  const roles = ["KC memory (B1)", "LH reward (B2)", "CX state dist (B3)", "DN learning (B4)"];
  insts.forEach((inst, i) => {
    const el = $(`#brain-card-${i}`);
    if (!el) return;
    $(".role", el).textContent = roles[i] || inst.id;
    const sig = inst.h_signal ?? 0;
    const sigEl = $(".sig", el);
    sigEl.textContent = (sig > 0 ? "+" : "") + sig.toFixed(3);
    sigEl.className = "sig " + (sig > 0 ? "up" : sig < 0 ? "down" : "");
    $(".horm-bar i", el).style.width = `${clamp((inst.hormones?.dopamine ?? 0) * 60, 4, 100)}%`;
    $(".dom-chip", el).textContent = `${inst.dominant} ${(inst.p_dominant * 100 || 0).toFixed(0)}%`;
  });
  // consensus canvas (tf1 history)
  drawSparkline($("#brain-consensus-chart"),
    (snap.brains?.consensus?.[`BTC|tf1`] || []).slice(-60), "#00e5ff");
  // learned influence weights (Section 10)
  const inf = snap.learning?.brain_influence || {};
  $("#brain-influence").innerHTML = TF.map(tf => {
    const w = inf[tf] || [0.25, 0.25, 0.25, 0.25];
    return `<div class="f-var"><span>${TF_LABEL[tf]}</span><b>${w.map(x => x.toFixed(3)).join(" · ")}</b></div>`;
  }).join("");
  if (brainsTab.ready && ACTIVE_TAB === "brains") refreshActivations(false);
}

function drawSparkline(cv, data, color) {
  if (!cv || !data?.length) return;
  const ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * 2, H = cv.height * 140;
  ctx.clearRect(0, 0, W, H);
  const min = Math.min(...data), max = Math.max(...data), rng = (max - min) || 1;
  ctx.strokeStyle = color || "#00e5ff"; ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - 10 - ((v - min) / rng) * (H - 20);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

/* ═══════════════════════════ FORMULA LAB ═══════════════════════════ */
const LATEX = {
  core: {
    1: "\\text{SATE} = \\frac{\\ln(T_{fix}+1)}{\\ln(T_{rel}+1)}",
    2: "\\text{VAD} = \\frac{P - \\text{VWAP}}{\\text{VWAP}} \\cdot \\text{RSI}_{z}",
    3: "\\text{WRPP} = \\frac{\\sum_i w_i (P_i - P_{i-1})}{\\sum_i P_{i-1}},\\ w_i = \\tfrac{i}{\\sum i}",
    4: "\\text{QZM} = \\left(\\frac{\\sigma_{fast}}{\\sigma_{slow}}\\right)^{\\phi},\\ \\phi = \\frac{\\text{RSI}}{100}",
    5: "\\text{SVAO} = \\frac{O - H_{prev}}{H_{prev} - L_{prev}}",
    6: "\\text{CETR} = \\sum_{i=0}^{k-1} |P_{-i} - P_{-i-1}| \\cdot (0.98)^i",
    7: "\\text{RVI} = \\frac{\\text{Corr}(\\Delta P, \\Delta V)}{1 + |\\Delta V|}",
    8: "\\text{NIV} = \\frac{\\text{BuyVol} - \\text{SellVol}}{\\text{BuyVol} + \\text{SellVol}}",
    9: "\\text{BAS} = \\frac{P_{bid}^{max} - P_{ask}^{min}}{P_{bid}^{max} + P_{ask}^{min}}",
    10: "\\text{IBDL} = \\frac{\\text{IDiv}_{bid} - \\text{IDiv}_{ask}}{\\text{IDiv}_{bid} + \\text{IDiv}_{ask}}",
    11: "\\text{MSRC} = \\text{Corr}(\\text{MSRC}_{lag1}, \\Delta P)",
    12: "\\text{TVF} = \\frac{\\sum_i V_i \\cdot |\\Delta P_i|}{\\sum_i V_i}",
    13: "\\text{OFA} = \\frac{\\sum_{\\Delta t < 5m} V_{buy} - \\sum V_{sell}}{\\sum_{24h} V}",
    14: "\\text{IVD} = \\frac{\\sigma^2_{intraday}}{\\sigma^2_{close}}",
    15: "\\text{DIQ} = \\frac{P - Q_{0.5}(P)}{Q_{0.95}(P) - Q_{0.05}(P)}",
    16: "\\text{CPCV} = 0.5\\big[\\text{Corr}(P,V)_{20} - \\text{Corr}(P,V)_{60}\\big]",
    17: "\\text{AOFI} = \\sum_{i=1}^{20} i \\cdot (B_i - S_i)",
    18: "\\text{LMPT} = \\frac{P_{close} - P_{open}}{P_{high} - P_{low}}",
    19: "\\text{CLRE} = \\frac{\\text{Corr}(\\Delta P, \\Delta\\text{LIQ})_{20}}{\\text{Corr}(\\Delta P, \\Delta\\text{LIQ})_{60}}",
    20: "\\text{DRHE} = e^{-\\lambda t},\\ \\lambda = \\lambda_0 \\cdot \\text{VIX}_{norm}",
    21: "\\text{CCI} = \\frac{\\sum_{i=1}^{n} \\omega_i F_i}{\\sum_i \\omega_i} \\cdot \\text{DRHE}",
    28: "\\text{ARC} = \\left(\\frac{\\text{ATR}_{cur}}{\\text{ATR}_{avg}}\\right)^{\\alpha} \\cdot \\text{conf}",
    29: "\\text{DLP} = \\alpha \\cdot \\text{ATR} \\cdot \\sqrt{T_{rem}}",
    30: "\\text{PCV} = \\frac{\\text{Corr}(\\text{CCI}_{f}, \\Delta P) - \\text{Corr}(\\text{CCI}_{l}, \\Delta P)}{\\sigma_{\\text{CCI}}}",
  },
  physics: {
    1: "P_1^{(j)} = \\frac{\\phi\\!\\left(\\frac{E_j - E_{th}}{\\sigma}\\right)}{\\sum_k \\phi\\!\\left(\\frac{E_k - E_{th}}{\\sigma}\\right)},\\ E_j = \\tfrac{1}{2}\\dot{x}_j^2 + V(x_j),\\ E_{th}=\\tfrac{1}{N}\\sum E_k",
    2: "\\sum_i (H_{ij} - E_i\\delta_{ij})\\psi_i = 0,\\ H_{ij}=\\tfrac{\\Phi_i\\Phi_j}{1+e^{-\\beta(E_i-E_j)}}",
    3: "P_3 = \\tanh\\!\\left(\\frac{\\Delta x}{\\lambda_D}\\right),\\ \\lambda_D=\\sqrt{\\tfrac{\\epsilon_0 k T}{n e^2}}",
    4: "G = \\left|\\frac{1}{N}\\sum_{k=1}^{N} e^{i\\phi_k}\\right|,\\ \\phi_k = \\tfrac{2\\pi P_k}{P_{ref}}",
    5: "P_5 = \\int_{T}^{T_{lock}} \\Gamma(t)\\,dt,\\ \\Gamma = \\Gamma_0 e^{-E_a/k_BT}",
    6: "DRPT = 1 - e^{-\\gamma\\tau},\\ \\gamma = \\frac{|\\text{MSRC}|}{\\sigma_{CTHR}},\\ CTHR=0.35",
    7: "P_7 = \\tanh\\!\\left(\\kappa\\,(T - T_c)\\right),\\ \\kappa=1000,\\ T_c = \\frac{T_{prev}+T_{next}}{2}",
    8: "P_8 = \\left(\\frac{T_{crit} - T}{T_{crit}}\\right)^{0.5},\\ T = \\frac{V_{high}+V_{low}}{2}",
    9: "P_9 = \\frac{1}{2}\\left[\\tanh\\!\\left(\\frac{E-E_c}{w}\\right) + 1\\right]",
    10: "K = \\frac{K_{max}}{1 + e^{-\\beta_{K}(\\Delta P - P_{50})}},\\ K_{max}=1000,\\ M=\\min(20,N)",
    11: "P_{11}^{(n)} = \\alpha P_{11}^{(n-1)} + \\beta\\,(1-\\alpha)\\sin(2\\pi n/\\phi)",
    12: "P_{12} = \\frac{T - T_c}{T_c},\\ T_c = \\frac{2}{\\pi}\\arctan\\!\\left(\\frac{V}{D}\\right)",
    13: "\\frac{d\\text{NE}}{dt} = k_{syn}\\,(P_{11}^{(n)} - P_{11}^{(n)}) - k_{deg}\\,\\text{NE}",
    14: "P_{14} = \\frac{P_{open}^{eff} - P_{close}^{eff}}{P_{open}^{eff} + P_{close}^{eff}}",
    15: "P_{15} = \\frac{1}{1 + e^{-\\beta(F_{11} - F_{11}^{crit})}}",
    16: "P_{16} = -\\partial^2 V_{eff}/\\partial P^2,\\ V_{eff}(P) = \\sum \\frac{(P-P_i)^2}{2\\sigma_i^2} - \\sum \\ln\\cosh",
    17: "P_{17} = \\left|\\frac{\\text{Im}(\\chi)}{\\text{Re}(\\chi)}\\right|,\\ \\chi = \\chi' + i\\chi''",
    18: "S_{ent} = -\\sum_j P_1^{(j)} \\ln P_1^{(j)}",
    19: "P_{19} = 1 - e^{-\\sigma^2 T/2}",
    20: "\\text{Weibull}(t) = \\frac{k}{\\lambda}\\left(\\frac{t}{\\lambda}\\right)^{k-1} e^{-(t/\\lambda)^k}",
    21: "\\tau = \\min_{i \\ne j} \\left\\{\\left|\\frac{x_i - x_j}{\\sigma_i - \\sigma_j}\\right| : \\text{corr}(x_i,x_j) > 0.5\\right\\}",
    22: "UFP = \\sum_{i=1}^{5} \\alpha_i P_i \\cdot \\text{coupling}_i,\\ \\sum \\alpha_i = 1",
  },
};

const EMO_ALIASES = ["FOMO", "Greed", "Fear", "Panic", "Hope", "Euphoria", "Anxiety",
  "Denial", "Regret", "Confusion", "Relief", "Excitement", "BoreDom", "Disgust",
  "Frustration", "Impatience", "Resignation", "Trust", "Doubt", "Cynicism",
  "Complacency", "CERC COMPOSITE"];

/* term-breakdown variable names per formula (values pulled live) */
const TERM_VARS = {
  1: ["T_fix", "T_rel", "ln ratio"], 2: ["P", "VWAP", "RSI_z", "dev"], 3: ["w_i", "ΣΔP", "WRPP"],
  4: ["σ_fast", "σ_slow", "φ", "QZM"], 5: ["Open", "H_prev", "L_prev", "SVAO"],
  6: ["Σ|ΔP|·0.98^k", "CETR"], 7: ["ρ(ΔP,ΔV)", "|ΔV|", "RVI"], 8: ["BuyVol", "SellVol", "NIV"],
  9: ["Bid_max", "Ask_min", "BAS"], 10: ["IDiv_bid", "IDiv_ask", "IBDL"],
  11: ["ρ(MSRC_lag1, ΔP)", "MSRC", "HMM γ"], 12: ["Σ V·|ΔP|", "ΣV", "TVF"],
  13: ["V_buy(<5m)", "V_sell(<5m)", "V_24h", "OFA"], 14: ["σ²_intraday", "σ²_close", "IVD"],
  15: ["P", "Q0.5", "Q0.95−Q0.05", "DIQ"], 16: ["ρ20(P,V)", "ρ60(P,V)", "CPCV"],
  17: ["i·(B_i−S_i)", "AOFI"], 18: ["Close", "Open", "High", "Low", "LMPT"],
  19: ["ρ20(ΔP,ΔLIQ)", "ρ60(ΔP,ΔLIQ)", "CLRE"], 20: ["λ", "λ0", "VIX_norm", "t", "DRHE"],
  21: ["ω_i", "ΣωF", "DRHE", "CCI"], 28: ["ATR_cur", "ATR_avg", "α", "conf", "ARC"],
  29: ["α", "ATR", "√T_rem", "DLP"], 30: ["ρ_f", "ρ_l", "σ_CCI", "PCV"],
};

let FORMULA_CACHE = { hist: null, systems: null, snapshot: null };

TAB_LOADERS.formulas = async () => {
  try {
    const [hist, sys, snap] = await Promise.all([
      api(`/api/formula-history?symbol=${dash.symbol}&tf=tf1`),
      api(`/api/systems?symbol=${dash.symbol}`),
      api("/api/snapshot"),
    ]);
    FORMULA_CACHE = { hist, systems: sys, snapshot: snap };
    renderFormulaCards();
  } catch (e) { toast("formula lab: " + e.message, "err"); }
};

function renderFormulaCards() {
  const q = ($("#formula-search")?.value || "").toLowerCase();
  const filter = $("#formula-filter")?.value || "all";
  const box = $("#formula-cards");
  const { hist, systems } = FORMULA_CACHE;
  const cards = [];
  const mkCore = f => {
    const name = hist?.names?.core?.[f] || `Formula ${f}`;
    const val = hist?.core?.[f]?.slice(-1)[0] ?? null;
    const w = hist?.weights?.[f];
    const accSeries = hist?.core_accuracy?.[f] || [];
    const acc = accSeries.length ? accSeries.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, accSeries.length) : null;
    cards.push({ sys: "core", id: `F${f}`, name, val, extra: `ω ${fmt(w, 4)}${acc != null ? ` · acc ${(acc > 0 ? "+" : "") + fmt(acc, 2)}` : ""}`, f, latex: LATEX.core[f] });
  };
  const mkEmo = e => {
    const name = `E${e} · ${EMO_ALIASES[e - 1] || ""}`;
    const val = hist?.emotion?.[e]?.slice(-1)[0] ?? null;
    cards.push({ sys: "emotion", id: `E${e}`, name, val, extra: hist?.emotion_weights?.[e] != null ? `ω ${fmt(hist.emotion_weights[e], 4)}` : "", f: `e${e}`, latex: null });
  };
  const mkPhys = p => {
    const name = hist?.names?.physics?.[p] || `P${p}`;
    const val = hist?.physics?.[p]?.slice(-1)[0] ?? null;
    cards.push({ sys: "physics", id: `P${p}`, name, val, extra: "", f: `p${p}`, latex: LATEX.physics[p] });
  };
  for (let f = 1; f <= 31; f++) mkCore(f);
  for (let e = 1; e <= 22; e++) mkEmo(e);
  for (let p = 1; p <= 22; p++) mkPhys(p);

  box.innerHTML = "";
  cards.forEach(c => {
    if (filter !== "all" && c.sys !== filter) return;
    if (q && !(`${c.id} ${c.name}`.toLowerCase().includes(q))) return;
    const el = document.createElement("div");
    el.className = "f-card";
    el.innerHTML = `
      <div class="f-head">
        <span class="f-id">${c.id}</span>
        <span class="f-name">${c.name}</span>
        <span class="f-val" style="color:${(c.val ?? 0) >= 0 ? "var(--cyan)" : "var(--red)"}">${fmt(c.val, 4)}</span>
        <span class="f-acc">${c.extra}</span>
        <span class="f-toggle">▸</span>
      </div>
      <div class="f-body">
        ${c.latex ? `<div class="f-latex" data-tex="${encodeURIComponent(c.latex)}"></div>` : ""}
        <div class="panel-h" style="margin:8px 0 6px">Term Breakdown — current values (per GRC)</div>
        <div class="f-breakdown" data-fid="${c.f}"></div>
        <div class="f-note">values refresh at each GRC boundary · weights learn every cycle (η rules)</div>
      </div>`;
    el.querySelector(".f-head").addEventListener("click", () => {
      el.classList.toggle("open");
      if (el.classList.contains("open")) {
        const texEl = el.querySelector(".f-latex");
        if (texEl && !texEl.dataset.rendered && typeof katex !== "undefined") {
          try { katex.render(decodeURIComponent(texEl.dataset.tex), texEl, { throwOnError: false }); texEl.dataset.rendered = "1"; } catch { /* keep raw */ }
        }
        fillTermBreakdown(el.querySelector(".f-breakdown"));
      }
    });
    box.appendChild(el);
  });
}

function fillTermBreakdown(el) {
  const fid = el.dataset.fid;
  const { hist, systems } = FORMULA_CACHE;
  const tf = "tf1";
  const rows = [];
  if (fid.startsWith("e")) {
    const e = Number(fid.slice(1));
    const sys = systems?.systems?.[`${dash.symbol}|${tf}`] || systems?.[`${dash.symbol}|${tf}`];
    const em = sys?.emotion || {};
    rows.push(["intensity E" + e, fmt(em.emotions?.[e], 4)]);
    rows.push(["direction", em.directions?.[e] ?? "—"]);
    rows.push(["CERC", fmt(em.cerc, 4)]);
    rows.push(["RTP", fmt(em.rtp, 4)]);
    rows.push(["TEDR", fmt(em.tedr, 4)]);
    if (em.weights) rows.push(["ω_E" + e, fmt(em.weights[e - 1], 5)]);
  } else if (fid.startsWith("p")) {
    const p = Number(fid.slice(1));
    const sys = systems?.systems?.[`${dash.symbol}|${tf}`] || {};
    const det = sys?.physics?.details || {};
    const d = det[`p${p}`] || det[p] || {};
    Object.entries(d).forEach(([k, v]) => {
      if (typeof v === "number") rows.push([k, fmt(v, 5)]);
      else if (Array.isArray(v) && v.length && typeof v[0] === "number") rows.push([k, `n=${v.length} μ=${fmt(v.reduce((a, b) => a + b, 0) / v.length, 4)}`]);
    });
    if (!rows.length) rows.push(["(physics details at GRC)", "—"]);
    rows.push(["UFP α", fmt(sys.physics?.alpha?.[p], 5)]);
    rows.push(["coupling", fmt(sys.physics?.couplings?.[p], 4)]);
  } else {
    const f = Number(fid);
    const name = TERM_VARS[f] || [];
    const fv = hist?.core?.[f]?.slice(-1)[0];
    const w = hist?.weights?.[f];
    const spark = hist?.core?.[f] || [];
    rows.push(["F (raw)", fmt(fv, 5)]);
    rows.push(["weight ω", fmt(w, 5)]);
    rows.push(["logic", FORMULA_LOGIC[Number(f)] || FORMULA_LOGIC[f] || "—"]);
    if (spark.length) {
      rows.push(["min(100)", fmt(Math.min(...spark), 4)]);
      rows.push(["max(100)", fmt(Math.max(...spark), 4)]);
      rows.push(["mean(100)", fmt(spark.reduce((a, b) => a + b, 0) / spark.length, 4)]);
    }
    name.forEach(n => rows.push([n, "live"]));
  }
  el.innerHTML = rows.map(([k, v]) => `<div class="f-var"><span>${k}</span><b>${v}</b></div>`).join("");
}
$("#formula-search")?.addEventListener("input", () => renderFormulaCards());
$("#formula-filter")?.addEventListener("change", () => renderFormulaCards());

/* ═══════════════════════════ EMOTION GRID ═══════════════════════════ */
TAB_LOADERS.emotion = async () => {
  try {
    const [sys, snap] = await Promise.all([
      api(`/api/systems?symbol=${dash.symbol}`),
      api("/api/snapshot"),
    ]);
    FORMULA_CACHE.systems = sys; FORMULA_CACHE.snapshot = snap;
    renderEmotionGrid();
  } catch (e) { toast("emotion: " + e.message, "err"); }
};

function renderEmotionGrid() {
  const tf = "tf1";
  const sys = FORMULA_CACHE.systems?.systems?.[`${dash.symbol}|${tf}`] || {};
  const em = sys.emotion || {};
  const grid = $("#emotion-grid");
  if (!grid.children.length) {
    for (let i = 1; i <= 22; i++) {
      const cell = document.createElement("div");
      cell.className = "e-cell";
      cell.id = `e-cell-${i}`;
      cell.innerHTML = `<div class="e-alias">${EMO_ALIASES[i - 1] || ""}</div>
        <div class="e-idx">E${i}</div>
        <div class="e-circle" data-val="0.00" style="--p:0%"></div>
        <span class="e-tag">NEUTRAL</span>`;
      grid.appendChild(cell);
    }
  }
  let sum = 0, cnt = 0;
  for (let i = 1; i <= 22; i++) {
    const v = em.emotions?.[i] ?? 0;
    const dir = em.directions?.[i] ?? 0;
    sum += Math.abs(v); cnt++;
    const cell = $(`#e-cell-${i}`);
    const circle = $(".e-circle", cell);
    circle.style.setProperty("--p", `${clamp(Math.abs(v) * 100, 0, 100)}%`);
    circle.dataset.val = (v >= 0 ? "+" : "") + v.toFixed(2);
    cell.classList.toggle("bearish", dir < 0);
    cell.classList.toggle("bullish", dir > 0);
    const tag = $(".e-tag", cell);
    const prev = Number(circle.dataset.prev || 0);
    let label = "NEUTRAL";
    if (Math.abs(v) > 0.7) label = "EXTREME";
    else if (Math.abs(v) > 0.45) label = "ELEVATED";
    else if (Math.abs(v) < Math.abs(prev) - 0.05) label = "DECAYING";
    tag.textContent = label;
    tag.className = `e-tag ${label}`;
    cell.classList.toggle("pulse", Math.abs(v) > 0.7);
    circle.dataset.prev = v;
  }
  const cerc = em.cerc ?? 0;
  const disp = cnt ? sum / cnt : 0;
  const narr = $("#emotion-narrative");
  if (narr) {
    const top = Object.entries(em.emotions || {})
      .map(([k, v]) => ({ k: +k, v, d: em.directions?.[k] ?? 0 }))
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 5);
    narr.innerHTML = `
      <div class="why-reason">${em.narrative || "Waiting for the first emotion pass…"}</div>
      <div class="contrib-row">${top.map(t => `
        <div class="contrib"><span class="cname">E${t.k} ${EMO_ALIASES[t.k - 1] || ""}</span>
          <div class="bar"><i class="${t.v >= 0 ? "pos" : "neg"}"
            style="width:${clamp(Math.abs(t.v) * 100, 2, 100)}%"></i></div>
          <span class="mono">${t.v >= 0 ? "+" : ""}${fmt(t.v, 3)}</span>
          <span class="dim">${t.d > 0 ? "▲ bullish" : t.d < 0 ? "▼ bearish" : "· neutral"}</span>
        </div>`).join("")}</div>
      <div class="kv-row">
        <div class="kv"><span>CERC composite (E22)</span><b>${fmt(cerc, 4)}</b></div>
        <div class="kv"><span>average intensity</span><b>${fmt(disp, 4)}</b></div>
        <div class="kv"><span>RTP trauma</span><b>${fmt(em.rtp ?? 0, 3)}${(em.rtp ?? 0) > 0.6 ? " ⚠ suppresses confidence" : ""}</b></div>
        <div class="kv"><span>TEDR → P6 link</span><b>${fmt(em.tedr ?? 0, 4)}</b></div>
      </div>
      <div class="hint">Each circle is one of the 22 spec emotion formulas; the ring shows |value|, colour shows direction. CERC (E22) is the composite that votes inside the CCI.</div>`;
  }
  let regime = "NEUTRAL";
  if (cerc > 0.4 && disp > 0.5) regime = "EUPHORIC";
  else if (cerc < -0.4 && disp > 0.5) regime = "PANICKED";
  else if (Math.abs(cerc) < 0.15 && disp < 0.35) regime = "COMPLACENT";
  else if (cerc > 0.1 && disp < 0.3) regime = "CONFLICTED";
  else if (disp > 0.55) regime = "EXHAUSTED";
  $("#cerc-regime").textContent = regime;
  $("#cerc-value").textContent = `CERC ${fmt(cerc, 4)}`;
  // weights list (21 CERC members)
  $("#emotion-weights").innerHTML = (em.weights || []).slice(0, 21).map((w, i) =>
    `<div class="f-var"><span>E${i + 1} ${EMO_ALIASES[i]}</span><b>${fmt(w, 5)}</b></div>`).join("");
  drawSparkline($("#cerc-chart"),
    FORMULA_CACHE.hist?.emotion?.["22"]?.slice(-80) || [], "#ffd600");
}

/* ═══════════════════════════ PHYSICS ENGINE ═══════════════════════════ */
TAB_LOADERS.physics = async () => {
  try {
    const [sys, candles] = await Promise.all([
      api(`/api/systems?symbol=${dash.symbol}`),
      api(`/api/candles?symbol=${dash.symbol}&tf=tf1`),
    ]);
    FORMULA_CACHE.systems = sys;
    renderPhysics(sys, candles);
  } catch (e) { toast("physics: " + e.message, "err"); }
  if (dash.latest) renderBgn(dash.latest);
};

function renderPhysics(sysWrap, candlesWrap) {
  const sys = sysWrap?.systems?.[`${dash.symbol}|tf1`] || {};
  const ph = sys.physics || {};
  const p1 = ph.p1 || {};
  const probs = p1.probs || [];
  const levels = p1.levels || [];
  const price = (candlesWrap?.bars || []).slice(-1)[0]?.c;
  const cyc = dash.cycles[`${dash.symbol}|tf1`];
  // probability distribution curve (P1 quantum levels × |c_n|²)
  const cv = $("#prob-curve");
  if (cv && probs.length) {
    const ctx = cv.getContext("2d");
    const W = cv.width = cv.clientWidth * 2, H = cv.height = 480;
    ctx.clearRect(0, 0, W, H);
    const maxP = Math.max(...probs);
    const lo = levels[0] ?? 0, hi = levels[levels.length - 1] ?? 1;
    const xOf = v => clamp(((v - lo) / ((hi - lo) || 1)) * W, 0, W);
    ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 2.5; ctx.beginPath();
    probs.forEach((p, i) => {
      const x = levels.length ? xOf(levels[i]) : (i / (probs.length - 1)) * W;
      const y = H - 30 - (p / (maxP || 1)) * (H - 60);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.lineTo(W, H - 30); ctx.lineTo(0, H - 30); ctx.closePath();
    ctx.fillStyle = "rgba(0,229,255,0.08)"; ctx.fill();
    // markers
    const marker = (val, color, label) => {
      if (val == null || !levels.length) return;
      const x = xOf(val);
      if (x < 0 || x > W) return;
      ctx.strokeStyle = color; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, H - 30); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color; ctx.font = "600 20px JetBrains Mono, monospace";
      ctx.fillText(label, x + 6, 26);
    };
    marker(price, "#e8e8f0", "P");
    if (cyc) {
      marker(cyc.sl, "#ff1744", "SL");
      (cyc.targets || []).forEach((t, i) => t.level && marker(t.level, "#00e5ff", `TP${i + 1}`));
    }
  }
  $("#prob-legend").innerHTML =
    `<span><i style="background:#00e5ff"></i>P1 quantum distribution (Σp=${fmt(probs.reduce((a, b) => a + b, 0), 3)})</span>
     <span><i style="background:#e8e8f0"></i>price</span>
     <span><i style="background:#ff1744"></i>SL</span>
     <span><i style="background:#00e5ff"></i>TP</span>`;
  $("#physics-p1").textContent = `P1 σq ${fmt(p1.sigma_q, 4)}`;
  // quantum state summary
  const qd = ph.details?.p21 || {};
  $("#quantum-summary").innerHTML = [
    ["P1 width σq", fmt(p1.sigma_q, 5)],
    ["WFCP", fmt(ph.wfcp, 5)],
    ["Tunneling P15", fmt(ph.details?.p15?.p15 ?? ph.details?.p15, 5)],
    ["Entanglement entropy S18", fmt(ph.details?.p18?.entropy, 5)],
    ["Decoherence τ (QDPC)", fmt(qd.tau, 5)],
    ["TME", fmt(ph.tme, 5)],
    ["DRPT", fmt(ph.drpt, 5)],
    ["GSPI", fmt(ph.gspi, 5)],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  // phase-space scatter
  const sc = $("#phase-scatter");
  if (sc && (candlesWrap?.bars || []).length > 5) {
    const ctx = sc.getContext("2d");
    const W = sc.width = sc.clientWidth * 2, H = sc.height = 520;
    ctx.clearRect(0, 0, W, H);
    const cs = (candlesWrap.bars || []).slice(-200);
    const rets = [], vols = [];
    for (let i = 1; i < cs.length; i++) {
      rets.push(cs[i].close / cs[i - 1].close - 1);
      vols.push((cs[i].high - cs[i].low) / cs[i].close);
    }
    const mr = rets.reduce((a, b) => a + Math.abs(b), 0) / rets.length || 1;
    const mv = vols.reduce((a, b) => a + b, 0) / vols.length || 1;
    rets.forEach((r, i) => {
      const x = W / 2 + (r / (mr * 4)) * (W / 2 - 20);
      const y = H - 20 - (vols[i] / (mv * 4)) * (H - 40);
      if (x < 0 || x > W || y < 0 || y > H) return;
      const dens = clamp(Math.abs(r) / (mr * 3), 0, 1);
      const hue = r >= 0 ? 186 : 348;
      ctx.fillStyle = `hsla(${hue}, 100%, ${55 + dens * 20}%, ${0.25 + dens * 0.6})`;
      ctx.beginPath(); ctx.arc(x, y, 3 + dens * 2.4, 0, Math.PI * 2); ctx.fill();
    });
    ctx.strokeStyle = "#23233a"; ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
  }
  // P-values list
  $("#physics-values").innerHTML = Object.entries(ph.values || {}).map(([k, v]) =>
    `<div class="f-var"><span>P${k}</span><b style="color:${v >= 0 ? "var(--cyan)" : "var(--red)"}">${fmt(v, 4)}</b></div>`).join("");
}

/* ═══════════════════════════ NEWS FEED ═══════════════════════════ */
const newsSel = $("#news-symbol");
ASSETS.forEach(s => {
  const o = document.createElement("option"); o.value = s; o.textContent = s;
  newsSel.appendChild(o);
});
newsSel.addEventListener("change", () => TAB_LOADERS.news?.());
TAB_LOADERS.news = async () => {
  try {
    const sym = newsSel.value || dash.symbol;
    const d = await api(`/api/news/${sym}`);
    const tb = $("#news-table tbody");
    tb.innerHTML = "";
    (d.items || []).forEach(it => {
      const tr = document.createElement("tr");
      const relClass = it.reliability >= 0.8 ? "rel-gold" : it.reliability >= 0.5 ? "rel-silver" : "rel-bronze";
      const relLabel = it.reliability >= 0.8 ? "GOLD" : it.reliability >= 0.5 ? "SILVER" : "BRONZE";
      const ageMin = Math.max(0, (Date.now() / 1000 - (it.ts || 0)) / 60);
      const cat = (d.categories?.[it.category] || {});
      const decayLeft = Math.max(0, (cat.duration || 60) - ageMin);
      tr.innerHTML = `
        <td class="mono">${new Date((it.ts || 0) * 1000).toLocaleTimeString()}</td>
        <td>${it.headline}</td>
        <td class="${relClass}">${relLabel} ${fmt(it.reliability, 2)}</td>
        <td>${it.category} <span class="mono" style="color:var(--gold)">×${fmt(cat.weight, 2)}</span></td>
        <td class="mono" style="color:${it.sentiment >= 0 ? "var(--cyan)" : "var(--red)"}">${(it.sentiment >= 0 ? "+" : "") + fmt(it.sentiment, 2)}</td>
        <td class="impact-cell"><div class="impact-bar"><i style="width:${clamp(Math.abs(it.impact || 0) * 100, 3, 100)}%"></i></div>${fmt(it.impact, 3)}</td>
        <td class="decay-tag">${decayLeft > 0 ? "decay " + Math.floor(decayLeft) + "m" : "expired"}</td>`;
      tb.appendChild(tr);
    });
    if (!(d.items || []).length) tb.innerHTML = `<tr><td colspan="7" style="color:var(--dim)">no items</td></tr>`;
    $("#news-meta").textContent = `${d.source || "sim"} · ${d.count ?? (d.items || []).length} items`;
  } catch (e) { toast("news: " + e.message, "err"); }
  if (dash.latest) renderHedge(dash.latest);
};

/* ═══════════════════════════ AGENT CONTROL ═══════════════════════════ */
TAB_LOADERS.agents = async () => {
  try {
    const [d, st] = await Promise.all([
      api("/api/agents/status"),
      api("/api/settings").catch(() => null),
    ]);
    renderAgents(d);
    // prefill the key form so the user sees what is already saved
    const s = st?.settings;
    if (s) {
      const map = { "key-gemini": "gemini_key", "key-backup1": "backup1_key",
                    "key-backup2": "backup2_key", "key-backup3": "backup3_key",
                    "key-newsapi": "newsapi_key", "key-neuprint": "neuprint_key" };
      for (const [id, k] of Object.entries(map)) {
        const el = $("#" + id);
        if (el && !el.value && s[k]) el.value = s[k];
      }
    }
  } catch (e) { toast("agents: " + e.message, "err"); }
};
function updateModeChip(mode) {
  const chip = $("#mode-chip");
  if (!chip) return;
  chip.textContent = mode === "drosophila" ? "DROSOPHILA · formula core"
    : mode === "local" ? "LOCAL MODEL"
    : mode === "cloud" ? "CLOUD AGENTS" : "…";
  chip.className = "mode-chip mode-" + (mode || "drosophila");
}

function renderAgents(d) {
  const slots = d.agents?.slots || [];
  updateModeChip(d.agents?.mode);
  // mode banner — agents are OPTIONAL; keyless = full Drosophila/formula core
  const mode = d.agents?.mode || "drosophila";
  const bm = $("#agent-mode");
  if (bm) {
    bm.className = "mode-banner mode-" + mode;
    bm.innerHTML = mode === "drosophila"
      ? `<b>DROSOPHILA MODE</b> — no API keys needed. The app runs fully on the ` +
        `4-instance Drosophila brain, 31 formulas, math &amp; physics. ` +
        `Paste any key below to add an <i>optional</i> cloud second opinion.`
      : mode === "local"
      ? `<b>LOCAL MODEL</b> — on-device inference active. Cloud keys are optional extras.`
      : `<b>CLOUD AGENTS</b> — ${slots.filter(s => s.status === "connected").length} slot(s) configured. ` +
        `The Drosophila brain + formulas still run every GRC regardless.`;
  }
  const titles = ["Gemini (primary)", "Backup 1 · Gemini", "Backup 2 · GPT-4o", "Backup 3 · Claude"];
  $("#agent-slots").innerHTML = slots.map((s, i) => `
    <div class="agent-slot">
      <h4>${titles[i] || s.status}<span class="st-LED st-${s.status}"></span></h4>
      <div class="key-mask">${s.key || "(no key)"}</div>
      <div class="mono" style="font-size:11px;color:var(--dim)">${s.status}</div>
      <div class="rate-bar"><i style="width:${s.limit ? clamp(s.usage / s.limit * 100, 0, 100) : 0}%"></i></div>
      <div class="mono" style="font-size:10px;color:var(--dim)">${s.usage ?? 0} / ${s.limit ?? "—"} calls this hour</div>
      ${s.last?.insight ? `<div class="hint">“${s.last.insight}”</div>` : ""}
      <button class="mini-btn test-slot" data-slot="${["gemini", "backup1", "backup2", "backup3"][i]}" style="margin-top:8px">Test Connection</button>
    </div>`).join("");
  const lo = d.agents?.local || {};
  $("#local-model-card").innerHTML = `
    <div class="f-var"><span>model</span><b>${lo.label || "—"}</b></div>
    <div class="f-var"><span>available</span><b>${lo.available ? "yes" : "no"}</b></div>
    <div class="f-var"><span>RAM tier</span><b>${lo.ram_gb ?? "?"} GB</b></div>
    <div class="f-var"><span>last inference</span><b>${lo.last_ms ? lo.last_ms + " ms" : "—"}</b></div>
    ${lo.error ? `<div class="hint" style="color:var(--amber)">${lo.error}</div>` : ""}
    <button class="mini-btn test-slot" data-slot="local" style="margin-top:8px">Test Connection</button>`;
  const sig = d.signal || {};
  $("#agent-signal-card").innerHTML = sig.source && sig.source !== "none"
    ? `<div class="f-var"><span>direction</span><b>${sig.direction > 0 ? "▲ confirm" : sig.direction < 0 ? "▼ contradict" : "0 uncertain"}</b></div>
       <div class="f-var"><span>confidence</span><b>${fmt(sig.confidence, 3)}</b></div>
       <div class="f-var"><span>weight</span><b>${fmt(sig.weight, 4)}</b></div>
       <div class="f-var"><span>source</span><b>${sig.source}${sig.fallback ? " (fallback)" : ""}</b></div>
       ${sig.insight ? `<div class="hint">“${sig.insight}”</div>` : ""}
       ${sig.warning ? `<div class="hint" style="color:var(--amber)">${sig.warning}</div>` : ""}`
    : `<div class="hint">No cloud signal — that is fine. The <b>Drosophila brain + formulas + physics</b> ` +
      `carry every vote (agents are optional, the 4-vote check still runs).</div>`;
  $$(".test-slot").forEach(b => b.addEventListener("click", async () => {
    b.textContent = "testing…";
    try {
      const r = await api(`/api/agents/test/${b.dataset.slot}`, { method: "POST" });
      toast(`${b.dataset.slot}: ${r.result?.ok ? "OK" : "FAILED — " + (r.result?.detail?.reason || r.result?.error || "")}`,
        r.result?.ok ? "ok" : "warn");
    } catch (e) { toast(e.message, "err"); }
    b.textContent = "Test Connection";
    TAB_LOADERS.agents();
  }));
  $("#led-agent").className = "led " + (slots[0]?.status === "connected" ? "on" : "off");
}
$("#agents-reload")?.addEventListener("click", () => TAB_LOADERS.agents());
$("#save-keys")?.addEventListener("click", async () => {
  const body = {
    gemini_key: $("#key-gemini").value.trim(),
    backup1_key: $("#key-backup1").value.trim(),
    backup2_key: $("#key-backup2").value.trim(),
    backup3_key: $("#key-backup3").value.trim(),
    newsapi_key: $("#key-newsapi").value.trim(),
    neuprint_key: $("#key-neuprint").value.trim(),
  };
  try {
    await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    toast("keys saved (masked)");
    TAB_LOADERS.agents();
  } catch (e) { toast(e.message, "err"); }
});

/* ═══════════════════════════ HISTORY ═══════════════════════════ */
const histState = { days: [], day: null, rows: [], i: 0, timer: null };
TAB_LOADERS.history = async () => {
  try {
    const [days, lb, snap] = await Promise.all([
      api("/api/history/days"), api("/api/history/leaderboard"), api("/api/snapshot"),
    ]);
    histState.days = days.days || [];
    renderCalendar();
    renderLeaderboard(lb.leaderboard || []);
    drawRolling(snap);
    $("#hist-meta").textContent = `${histState.days.length} days · retention 90d → weekly`;
    if (histState.days.length) openDay(histState.days[0].date);
  } catch (e) { toast("history: " + e.message, "err"); }
};
function renderCalendar() {
  const box = $("#history-calendar");
  box.innerHTML = "";
  const row = document.createElement("div");
  row.className = "cal-row";
  histState.days.forEach(d => {
    const cls = d.accuracy == null ? "n" : d.accuracy > 0.8 ? "g" : d.accuracy >= 0.6 ? "y" : "r";
    const el = document.createElement("div");
    el.className = `cal-day ${cls}` + (histState.day === d.date ? " sel" : "");
    el.textContent = `${d.date.slice(5)}\n${d.accuracy != null ? Math.round(d.accuracy * 100) + "%" : "—"}`;
    el.style.whiteSpace = "pre";
    el.title = `${d.date}: ${d.grcs} GRCs, acc ${d.accuracy != null ? (d.accuracy * 100).toFixed(1) + "%" : "—"}`;
    el.addEventListener("click", () => openDay(d.date));
    row.appendChild(el);
  });
  box.appendChild(row);
}
function renderLeaderboard(rows) {
  $("#leaderboard tbody").innerHTML = rows.slice(0, 12).map((r, i) => `
    <tr><td class="mono">${i + 1}</td><td>F${r.formula} ${r.name || ""}</td>
    <td class="mono" style="color:${r.accuracy > 0.6 ? "var(--green)" : r.accuracy > 0.5 ? "var(--gold)" : "var(--red)"}">${(r.accuracy * 100).toFixed(1)}%</td></tr>`).join("");
}
function drawRolling(snap) {
  const tf = "tf1";
  const w = snap.learning?.tf_hr || {};
  const cv = $("#rolling-chart");
  const ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * 2, H = cv.height = 360;
  ctx.clearRect(0, 0, W, H);
  const names = TF;
  names.forEach((t, i) => {
    const v = clamp(w[t] ?? 0.5, 0, 1);
    const x = 60 + i * (W - 100) / 4, bw = (W - 120) / 6;
    const h = v * (H - 60);
    ctx.fillStyle = v > 0.85 ? "#2ee6a8" : v >= 0.6 ? "#ffd600" : "#ff1744";
    ctx.fillRect(x, H - 30 - h, bw, h);
    ctx.fillStyle = "#8a8aa8"; ctx.font = "20px JetBrains Mono, monospace";
    ctx.fillText(TF_LABEL[t].split(" ")[0], x, H - 8);
    ctx.fillStyle = "#e8e8f0"; ctx.fillText((v * 100).toFixed(0) + "%", x + 8, H - 40 - h);
  });
}
async function openDay(date) {
  try {
    const d = await api(`/api/history/day?date=${date}`);
    histState.day = date; histState.rows = d.records || []; histState.i = 0;
    $("#day-detail-date").textContent = date;
    renderCalendar();
    renderDayTable();
    renderReplay();
  } catch (e) { toast(e.message, "err"); }
}
function renderDayTable() {
  $("#day-table tbody").innerHTML = histState.rows.map((r, i) => {
    const o = Object.values(r.outcomes || {});
    const hits = o.filter(x => x.target_hit).length;
    const sls = o.filter(x => x.sl_hit).length;
    return `<tr style="cursor:pointer" data-i="${i}">
      <td class="mono">${(r.timestamp || "").slice(11, 19)}</td>
      <td class="mono">${fmt(r.prices?.BTC, 1)}</td>
      <td class="mono" style="color:${hits ? "var(--green)" : sls ? "var(--red)" : "var(--dim)"}">${hits}T ${sls}S</td></tr>`;
  }).join("");
  $$("#day-table tr").forEach(tr => tr.addEventListener("click", () => {
    histState.i = Number(tr.dataset.i); renderReplay();
  }));
}
function renderReplay() {
  const r = histState.rows[histState.i];
  $("#replay-pos").textContent = `${histState.i + 1} / ${histState.rows.length}`;
  if (!r) { $("#replay-body").textContent = "no records"; return; }
  $("#replay-body").innerHTML = `
    <div><b>GRC @ ${new Date(r.timestamp).toLocaleTimeString()}</b></div>
    ${Object.entries(r.predictions || {}).map(([tf, p]) => {
      const o = r.outcomes?.[tf] || {};
      const dir = p.direction > 0 ? "up" : p.direction < 0 ? "down" : "";
      return `<div>${TF_LABEL[tf]}:
        <span class="rb-dir ${dir}">${p.direction > 0 ? "▲" : p.direction < 0 ? "▼" : "●"}</span>
        conf ${fmt(p.confidence, 3)} · ${p.issued ? "ISSUED" : "NO TRADE"}
        ${o.target_hit ? " · <span style='color:var(--green)'>TP HIT</span>" : ""}
        ${o.sl_hit ? " · <span style='color:var(--red)'>SL HIT</span>" : ""}</div>`;
    }).join("")}`;
}
$("#replay-prev")?.addEventListener("click", () => { histState.i = Math.max(0, histState.i - 1); renderReplay(); });
$("#replay-next")?.addEventListener("click", () => { histState.i = Math.min(histState.rows.length - 1, histState.i + 1); renderReplay(); });
$("#replay-play")?.addEventListener("click", () => {
  if (histState.timer) { clearInterval(histState.timer); histState.timer = null; $("#replay-play").textContent = "▶ play"; return; }
  $("#replay-play").textContent = "⏸ pause";
  histState.timer = setInterval(() => {
    if (histState.i >= histState.rows.length - 1) { clearInterval(histState.timer); histState.timer = null; $("#replay-play").textContent = "▶ play"; return; }
    histState.i++; renderReplay();
  }, 900);
});

/* ═══════════════════════════ SETTINGS ═══════════════════════════ */
TAB_LOADERS.settings = async () => {
  const snap = FORMULA_CACHE.snapshot || await api("/api/snapshot").catch(() => ({}));
  $("#learning-summary").textContent =
    `rates η_micro ${snap.learning?.rates?.micro} · shares ` +
    Object.entries(snap.learning?.subsystem_shares || {}).map(([k, v]) => `${k} ${v}`).join(", ");
  // assets / tfs checkboxes (UI filtering stored server-side)
  const sa = $("#set-assets"), st = $("#set-tfs");
  if (!sa.children.length) {
    sa.innerHTML = `<span style="font-size:11px;color:var(--dim)">assets</span>` +
      ASSETS.map(s => `<label class="switch"><input type="checkbox" class="set-asset" value="${s}" checked><span>${s}</span></label>`).join("");
    st.innerHTML = `<span style="font-size:11px;color:var(--dim)">timeframes</span>` +
      TF.map(t => `<label class="switch"><input type="checkbox" class="set-tf" value="${t}" checked><span>${TF_LABEL[t]}</span></label>`).join("");
  }
  // manual triggers
  const mt = $("#manual-triggers");
  if (!mt.children.length) {
    mt.className = "trigger-grid";
    ASSETS.forEach(s => TF.forEach(t => {
      const b = document.createElement("button");
      b.className = "mini-btn"; b.textContent = `${s} ${t.toUpperCase()}`;
      b.addEventListener("click", async () => {
        try { await api(`/api/override/${s}/${t}`, { method: "POST" }); toast(`${s} ${t}: fresh cycle locked`); haptic("lock"); }
        catch (e) { toast(e.message, "err"); }
      });
      mt.appendChild(b);
    }));
  }
};
$("#set-haptics")?.addEventListener("change", e => HAPTICS = e.target.checked);
$("#set-sound")?.addEventListener("change", e => SOUND = e.target.checked);
$("#reset-learning")?.addEventListener("click", async () => {
  if (!confirm("RESET ALL learned weights to the uniform prior? This cannot be undone.")) return;
  try { await api("/api/learning/reset", { method: "POST" }); toast("learning state reset"); }
  catch (e) { toast(e.message, "err"); }
});
$("#export-state")?.addEventListener("click", async () => {
  try {
    const d = await api("/api/state/export");
    const blob = new Blob([JSON.stringify(d.state, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "learning_state.json"; a.click();
  } catch (e) { toast(e.message, "err"); }
});
$("#import-state")?.addEventListener("click", () => $("#import-file").click());
$("#import-file")?.addEventListener("change", async e => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const state = JSON.parse(await f.text());
    await api("/api/state/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ state }) });
    toast("state imported + reloaded");
  } catch (err) { toast(err.message, "err"); }
});
/* asset / timeframe checkboxes filter the dashboard live */
document.addEventListener?.("change", () => {});
$("#set-assets")?.addEventListener?.("change", () => {});
document.addEventListener("change", e => {
  const t = e.target;
  if (t.classList?.contains("set-asset")) {
    const show = $$(".set-asset").filter(x => x.checked).map(x => x.value);
    $$(".asset-btn").forEach(b => { b.style.display = show.includes(b.textContent) ? "" : "none"; });
    if (!show.includes(dash.symbol) && show.length) {
      dash.symbol = show[0];
      loadCandles();
    }
  }
  if (t.classList?.contains("set-tf")) {
    const show = $$(".set-tf").filter(x => x.checked).map(x => x.value);
    TF.forEach(tf => {
      const q = $(`#q-${tf}`);
      if (q) q.style.display = show.includes(tf) ? "" : "none";
    });
  }
});

/* persist interface toggles */
async function persistSettings() {
  try {
    await api("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ haptics: HAPTICS, sound: SOUND }),
    });
  } catch { /* non-critical */ }
}
$("#set-haptics")?.addEventListener("change", persistSettings);
$("#set-sound")?.addEventListener("change", persistSettings);

/* ═══════════════════════════ WEBSOCKET + BOOT ═══════════════════════════ */
let ws, wsRetry = 0;
let lastMsgAt = 0;

function applySnapshot(snap) {
  // overlay FIRST — a dashboard render error must never freeze the 4:00 page
  try { updateOverlay(null, snap); } catch { /* overlay ids missing */ }
  try { renderDashboard(snap); } catch (e) { console.error("renderDashboard", e); }
  if (ACTIVE_TAB === "brains") { try { updateBrains(snap); } catch { /* tab */ } }
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { wsRetry = 0; $("#led-ws").className = "led on"; };
  ws.onclose = () => {
    $("#led-ws").className = "led off";
    setTimeout(connectWS, Math.min(8000, 1000 * ++wsRetry));
  };
  ws.onmessage = ev => {
    lastMsgAt = Date.now();
    const msg = JSON.parse(ev.data);
    handleMessage(msg);
  };
}
// ── Cross-part sync panels (Part 1–4 wiring restored) ───────────────
function handleTick(msg) {
  const price = msg.prices && msg.prices[dash.symbol];
  if (!price) return;
  dash.prices = msg.prices;
  const aset = dash.latest?.assets?.[dash.symbol];
  TF.forEach(tf => {
    const pe = $("#q-price-" + tf);
    if (!pe) return;
    pe.innerHTML = `<b style="color:var(--gold)">${fmt(price, 2)}</b>` +
      (aset ? ` <span style="color:${aset.window_change_pct >= 0 ? "var(--cyan)" : "var(--red)"}">` +
        `${aset.window_change_pct >= 0 ? "+" : ""}${aset.window_change_pct}%</span>` : "");
  });
  // live line: every tick redraws the canvas (direct Binance/sim feed link)
  TF.forEach(tf => {
    try { dash.charts[tf]?.tick(price); } catch { /* chart not ready */ }
  });
}

function renderCorel(snap) {
  const cv = $("#corel-canvas");
  const meta = $("#corel-meta");
  if (!cv || !meta) return;
  const sys = snap?.systems?.[`${dash.symbol}|tf1`];
  const corel = sys?.corelation;
  if (!corel || !corel.edges) { meta.textContent = "warming up…"; return; }
  const nodes = corel.nodes || [];
  const edges = corel.edges || [];
  cv.width = cv.clientWidth * 2 || 720; cv.height = 600;
  const g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  const pos = {};
  nodes.forEach((n, i) => {
    const a = (i / Math.max(nodes.length, 1)) * 2 * Math.PI;
    pos[n.id] = { x: cv.width / 2 + Math.cos(a) * (cv.width * 0.38),
                  y: cv.height / 2 + Math.sin(a) * (cv.height * 0.36) };
  });
  const colors = { brain: "#ffb02e", emotion: "#e879f9", physics: "#5b8cff", core: "#00e5ff", agent: "#2ee6a8" };
  edges.forEach(e => {
    const a = pos[e.a], b = pos[e.b];
    if (!a || !b) return;
    g.strokeStyle = (e.corr ?? 0) >= 0 ? "rgba(46,230,168,.75)" : "rgba(255,23,68,.75)";
    g.lineWidth = 1 + Math.abs(e.corr ?? 0) * 5;
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
  });
  nodes.forEach(n => {
    const p = pos[n.id]; if (!p) return;
    g.fillStyle = colors[n.kind] || "#00e5ff";
    g.beginPath(); g.arc(p.x, p.y, 14, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#fff"; g.font = "bold 18px sans-serif"; g.textAlign = "center";
    g.fillText(n.short || n.id.slice(0, 2), p.x, p.y + 6);
  });
  const mags = edges.map(e => Math.abs(e.corr ?? 0)).filter(v => v > 0);
  meta.textContent = `${nodes.length} nodes · ${edges.length} edges · avg |corr| ` +
    (mags.length ? fmt(mags.reduce((a, b) => a + b, 0) / mags.length, 3) : "0") +
    (corel.window ? ` · ${corel.window}-GRC window` : "");
}

function renderWiring(snap) {
  const el = $("#wiring-panel"); if (!el) return;
  const sys = snap?.systems?.[`${dash.symbol}|tf1`];
  const br = snap?.brain || {};
  const em = snap?.emotions || {};
  const ph = snap?.physics || {};
  const ls = snap?.learning || {};
  const hf = sys?.hedge || {};
  const fvs = (snap?.formula_values || {})[`${dash.symbol}|tf1`] || {};
  const active = Object.values(fvs).filter(v => Math.abs(v?.raw ?? 0) > 0.7).length;
  const dc = sys?.double_check;
  el.innerHTML = `
    <div class="kv"><span>affinity cells live (|F|&gt;0.7, tf1)</span><span class="val">${active}</span></div>
    <div class="kv"><span>CIRC → B2 experience</span><span class="val">${fmt(br.circ, 0)} / ${br.experience ?? 0}</span></div>
    <div class="kv"><span>TEDR → P6</span><span class="val">${fmt(em.tedr, 4)} → ${fmt(ph.drpt, 2)}Ω</span></div>
    <div class="kv"><span>TME → emotion share</span><span class="val">${fmt(em.tme, 4)} → ${fmt((ls.shares || {}).emotion, 4)}</span></div>
    <div class="kv"><span>HF flags (IFR/DPAD/SMDI)</span><span class="val">${hf.ifr ? "IFR " : ""}${hf.dpad ? "DPAD " : ""}${hf.smdi ? "SMDI " : ""}${(!hf.ifr && !hf.dpad && !hf.smdi) ? "—" : ""}</span></div>
    <div class="kv"><span>4-vote double-check</span><span class="val">${dc ? dc.label + (dc.agree ? " ✓" : " ✗") : "—"}</span></div>`;
}

function renderHedge(snap) {
  const el = $("#hedge-panel"); if (!el) return;
  const hf = snap?.systems?.[`${dash.symbol}|tf1`]?.hedge || {};
  if (!hf.combined) { el.innerHTML = "warming up… (IFR·DPAD·SMDI appear after the first HF1–HF3 evaluation)"; return; }
  const pc = hf.price_checks || [];
  const events = hf.events || [];
  const prints = hf.prints || [];
  el.innerHTML = `
    <div class="kv"><span>IFR (order-flow imbalance)</span><span class="val ${hf.ifr ? "flag-on" : ""}">${hf.ifr ? "FIRED" : "off"}</span></div>
    <div class="kv"><span>DPAD (dark-pool exhaustion)</span><span class="val ${hf.dpad ? "flag-on" : ""}">${hf.dpad ? "FIRED" : "off"}</span></div>
    <div class="kv"><span>SMDI (smart-money divergence)</span><span class="val ${hf.smdi ? "flag-on" : ""}">${hf.smdi ? "FIRED" : "off"}</span></div>
    <div class="kv"><span>HF combined</span><span class="val">${hf.combined}</span></div>
    ${hf.ifr ? `<div class="bar" title="IFR raw"><span style="width:${Math.min(100, Math.abs(hf.raw_ifr || 0) * 200)}%"></span></div>` : ""}
    ${hf.dpad ? `<div class="bar" title="DPAD raw"><span style="width:${Math.min(100, Math.abs(hf.raw_dpad || 0) * 200)}%"></span></div>` : ""}
    <div class="kv"><span>price checks / divergence</span><span class="val">${pc.length} / ${fmt(hf.divergence, 4)}</span></div>
    <div class="kv"><span>events (recent) / prints</span><span class="val">${events.length} / ${prints.length}</span></div>
    ${pc.slice(-4).reverse().map(p => {
      const c = p.price_corr || {};
      return `<div class="insight">· ${p.src}: ${p.verdict}` +
        (c.strength !== null && c.strength !== undefined ? ` <span class="mono">[${c.strength}]</span>` : "") + `</div>`;
    }).join("")}
    ${events.slice(-3).reverse().map(e => `<div class="insight">· ${e.name}</div>`).join("")}`;
}

function renderBgn(snap) {
  const el = $("#bgn-matrix"); if (!el) return;
  const b = snap?.bgn || {};
  if (!Object.keys(b).length) { el.innerHTML = "warming up…"; return; }
  el.innerHTML = `<div class="bgn-grid">` +
    `<div class="f-var">β</div>` + TF.map(t => `<div class="f-var">${t}</div>`).join("") +
    TF.flatMap(t => [`<div class="f-var">${t}</div>`,
      ...TF.map(t2 => {
        const v = b[t]?.[t2] ?? 0;
        return `<div class="f-var" style="color:${v >= 0 ? "var(--cyan)" : "var(--red)"}">${fmt(v, 3)}</div>`;
      })]).join("") + `</div>`;
}

function syncActivePanels(snap) {
  if (ACTIVE_TAB === "brains") { renderCorel(snap); renderWiring(snap); }
  else if (ACTIVE_TAB === "news") renderHedge(snap);
  else if (ACTIVE_TAB === "physics") renderBgn(snap);
}

function handleMessage(msg) {
  switch (msg.type) {
    case "snapshot":
      dash.lastGrc = msg.grc?.count ?? dash.lastGrc;
      // overlay first: a quadrant render error cannot freeze the analysis page
      if (msg.phase === "analysis") { try { updateOverlay(msg, msg); } catch { /* ids */ } }
      else $("#analysis-overlay").classList.add("gone");
      try { renderDashboard(msg); } catch (e) { console.error("renderDashboard", e); }
      if (ACTIVE_TAB === "brains") updateBrains(msg);
      syncActivePanels(msg);
      break;
    case "tick":
      handleTick(msg);
      break;
    case "phase":
      updateOverlay(msg, null);
      break;
    case "sl_event":
      haptic(msg.kind === "close" ? "close" : "sl");
      toast(msg.message || msg.kind, msg.kind === "close" ? "err" : "warn");
      break;
    case "event":
      if (msg.kind === "hit" || msg.kind === "target_hit") { haptic("tp"); beep(1040, 80); }
      break;
    case "ping": break;
  }
}

/* boot */
(async function boot() {
  connectWS();
  buildCharts();
  loadCandles();
  // initial REST fetch — applies phase too, so the 4-minute overlay
  // updates and lifts even before (or without) the WebSocket
  api("/api/snapshot").then(applySnapshot).catch(() => {});
  // REST fallback: if no WS message for ~4 s, poll the snapshot so the
  // countdown, dashboard and overlay keep working without WebSocket
  setInterval(async () => {
    if (Date.now() - lastMsgAt < 4000) return;
    try { applySnapshot(await api("/api/snapshot")); } catch { /* retry */ }
  }, 2000);
  // local 1 s countdown — the clock always moves, even between REST polls
  setInterval(() => {
    const ov = $("#analysis-overlay");
    if (!ov || ov.classList.contains("gone") || !window.__ovDeadline) return;
    const rem = Math.max(0, (window.__ovDeadline - Date.now()) / 1000);
    $("#analysis-clock").textContent =
      `${Math.floor(rem / 60)}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
  }, 1000);
  // quadrant rings/clocks tick every second (not per GRC)
  setInterval(updateCountdowns, 1000);
  // periodic refreshes for lazy tabs
  setInterval(() => {
    if (ACTIVE_TAB === "news") TAB_LOADERS.news?.();
    if (ACTIVE_TAB === "agents") TAB_LOADERS.agents?.();
    if (ACTIVE_TAB === "emotion") TAB_LOADERS.emotion?.();
    if (ACTIVE_TAB === "physics") TAB_LOADERS.physics?.();
    if (ACTIVE_TAB === "formulas") TAB_LOADERS.formulas?.();
    if (ACTIVE_TAB === "history") TAB_LOADERS.history?.();
    if (ACTIVE_TAB === "brains" && dash.latest) { renderCorel(dash.latest); renderWiring(dash.latest); }
  }, 20000);
})();
