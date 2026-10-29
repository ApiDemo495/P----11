/* Crypto Prediction 4 — dashboard runtime
 *
 * Contract:
 *  - prediction values arrive only in `snapshot` messages (GRC cadence and
 *    spec-driven events); between them only `tick` price messages stream.
 *  - countdowns render locally each second from cycle_end timestamps.
 *  - SL events trigger haptic vibration + alert sound per spec.
 */
"use strict";

const TF_ORDER = ["tf1", "tf2", "tf3", "tf4"];
const TF_LABEL = { tf1: "TF1 · 22m", tf2: "TF2 · 44m", tf3: "TF3 · 130m", tf4: "TF4 · 247m" };
const SYMBOLS = ["BTC", "PAXG", "ETH", "BNB"];

const state = {
  snap: null,
  ws: null,
  connected: false,
  fmTf: { BTC: "tf1", PAXG: "tf1", ETH: "tf1", BNB: "tf1" },
  detail: { sym: "BTC", tf: "tf1" },   // drives brain / emotion / risk panels
  history: { BTC: [], PAXG: [], ETH: [], BNB: [] },  // [ts, price] for sparklines
  bannerTimer: null,
};

const EMOTION_META = {
  1: ["FII", "Fear Intensity"], 2: ["GSL", "Guardian Loss"], 3: ["FPR", "Fear Put"],
  4: ["PCP", "Panic Coeff"], 5: ["CD", "Octo Reflex"], 6: ["ESi", "Endorphin"],
  7: ["DPI", "Dopamine"], 8: ["ATR", "Vol Climax"], 9: ["RTP", "Trauma"],
  10: ["ABS", "Absorption"], 11: ["HMC", "Herd Mirror"], 12: ["DRD", "Range Dev"],
  13: ["ERI", "Extreme Rev"], 14: ["OFS", "Opt/Fear"], 15: ["IAF", "Intent/Act"],
  16: ["LAI", "Lead Impulse"], 17: ["CDP", "Crowd Pull"], 18: ["HTI", "Trend Inv"],
  19: ["SMDM", "Smart Diverge"], 20: ["NMS", "News Surge"], 21: ["TEDR", "Trend Exh"],
  22: ["CERC", "Composite"],
};

/* ======================= formatting ======================= */
function fmtPrice(p) {
  if (p == null) return "—";
  if (p >= 10000) return p.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  if (p >= 100) return p.toFixed(2);
  return p.toFixed(4);
}
function fmtMMSS(sec) {
  sec = Math.max(0, Math.floor(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function fmtPct(x) { return (x >= 0 ? "+" : "") + x.toFixed(2) + "%"; }

/* ======================= sound / haptics ======================= */
let audioCtx = null;
function alertSound(kind) {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const now = audioCtx.currentTime;
    if (kind === "close") {
      // harsh double-tone alarm
      [880, 660, 880, 660].forEach((f, i) => {
        const o = audioCtx.createOscillator(), g = audioCtx.createGain();
        o.type = "square"; o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, now + i * 0.14);
        g.gain.exponentialRampToValueAtTime(0.25, now + i * 0.14 + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.14 + 0.12);
        o.connect(g).connect(audioCtx.destination);
        o.start(now + i * 0.14); o.stop(now + i * 0.14 + 0.13);
      });
    } else {
      // soft amber chime
      [523, 784].forEach((f, i) => {
        const o = audioCtx.createOscillator(), g = audioCtx.createGain();
        o.type = "sine"; o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, now + i * 0.18);
        g.gain.exponentialRampToValueAtTime(0.18, now + i * 0.18 + 0.03);
        g.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.18 + 0.35);
        o.connect(g).connect(audioCtx.destination);
        o.start(now + i * 0.18); o.stop(now + i * 0.18 + 0.4);
      });
    }
  } catch (e) { /* audio unavailable */ }
}
function haptic() {
  try { if (navigator.vibrate) navigator.vibrate([200, 80, 200, 80, 320]); } catch (e) {}
}

/* ======================= building the grid ======================= */
function buildGrid() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const sym of SYMBOLS) {
    const card = document.createElement("div");
    card.className = "asset-card";
    card.id = `card-${sym}`;
    card.innerHTML = `
      <div class="asset-head">
        <span class="asset-sym">${sym}</span>
        <span class="asset-price" id="price-${sym}">—</span>
        <span class="asset-chg" id="chg-${sym}">—</span>
        <span class="asset-mode" id="mode-${sym}">—</span>
      </div>
      <div class="spark-wrap"><canvas id="spark-${sym}"></canvas></div>
      <div class="news-strip" id="news-${sym}"></div>
      <div class="tf-grid">
        ${TF_ORDER.map(tf => tfPanelHtml(sym, tf)).join("")}
      </div>`;
    grid.appendChild(card);
  }
  // wire override buttons
  for (const sym of SYMBOLS) {
    for (const tf of TF_ORDER) {
      const btn = document.getElementById(`ovr-${sym}-${tf}`);
      if (btn) btn.addEventListener("click", (e) => {
        e.stopPropagation();
        fetch("/api/snapshot"); // warm
        fetch(`/api/override/${sym}/${tf}`, { method: "POST" })
          .catch(() => {});
      });
    }
  }
}

function tfPanelHtml(sym, tf) {
  return `
  <div class="tf" id="tf-${sym}-${tf}">
    <div class="tf-top">
      <span class="tf-name">${TF_LABEL[tf]}</span>
      <span class="tf-count" id="cnt-${sym}-${tf}">--:--</span>
    </div>
    <div class="tf-dir flat" id="dir-${sym}-${tf}">◆ awaiting</div>
    <div class="tf-conf-row">
      <div class="conf-bar"><div class="conf-fill" id="confb-${sym}-${tf}" style="width:0%"></div></div>
      <span class="conf-num" id="confn-${sym}-${tf}">—</span>
    </div>
    <div class="targets" id="tgts-${sym}-${tf}"></div>
    <div class="risk-row" id="risk-${sym}-${tf}">
      <span class="tp">TP —</span><span class="sl">SL —</span><span class="ratio-badge">—</span>
    </div>
    <button class="override-btn" id="ovr-${sym}-${tf}" title="Force immediate fresh cycle">↻ FRESH</button>
    <div class="no-trade hidden" id="nt-${sym}-${tf}">NO TRADE<br/>INSUFFICIENT CONFIDENCE</div>
    <div class="pending-reset hidden" id="pr-${sym}-${tf}">POSITION CLOSED<br/>next GRC resets</div>
    <div class="warn-flag hidden" id="warn-${sym}-${tf}">⚠ LOW-CONFIDENCE LOCK</div>
  </div>`;
}

/* ======================= rendering ======================= */
function renderPhase(msg) {
  const overlay = document.getElementById("analysis-overlay");
  const phaseLabel = document.getElementById("phase-label");
  if (msg.phase === "analysis") {
    overlay.classList.remove("hidden");
    phaseLabel.textContent = "ANALYSIS";
    document.getElementById("analysis-countdown").textContent = msg.remaining_s;
    document.querySelectorAll("#readiness-list li").forEach(li => {
      const k = li.dataset.k;
      li.classList.toggle("ok", !!(msg.readiness && msg.readiness[k]));
    });
  } else {
    overlay.classList.add("hidden");
    phaseLabel.textContent = msg.warning ? "LIVE · WARNED" : "LIVE";
    if (msg.warning) {
      showBanner("close", "⚠ INITIAL ANALYSIS EXPIRED",
        "Locked best available prediction — confidence warning flag active");
    }
  }
}

function showBanner(kind, title, detail) {
  const el = document.getElementById("event-banner");
  el.className = `event-banner ${kind}`;
  el.innerHTML = `${title}<span class="detail">${detail || ""}</span>`;
  el.classList.remove("hidden");
  clearTimeout(state.bannerTimer);
  state.bannerTimer = setTimeout(() => el.classList.add("hidden"),
    kind === "close" ? 9000 : 6500);
}

function renderSnapshot(snap) {
  state.snap = snap;
  document.getElementById("feed-label").textContent = "feed: " + (snap.assets.BTC?.mode || "—");
  document.getElementById("grc-meta").textContent =
    `cycle #${snap.grc.count} · id ${snap.grc.id}`;

  if (snap.phase === "analysis") renderPhase(snap.analysis ? { ...snap.analysis, phase: "analysis" } : { phase: "live" });
  else renderPhase({ phase: "live", warning: snap.analysis && snap.analysis.warning });

  for (const sym of SYMBOLS) {
    const a = snap.assets[sym];
    if (!a) continue;
    document.getElementById(`price-${sym}`).textContent = fmtPrice(a.price);
    const chg = document.getElementById(`chg-${sym}`);
    chg.textContent = fmtPct(a.window_change_pct);
    chg.className = "asset-chg " + (a.window_change_pct >= 0 ? "up" : "down");
    const mode = document.getElementById(`mode-${sym}`);
    mode.textContent = a.mode === "binance" ? "● LIVE BINANCE" : "◌ SIM FEED";
    mode.className = "asset-mode" + (a.mode === "binance" ? " live" : "");

    // push price into sparkline history
    if (a.price > 0) {
      const h = state.history[sym];
      h.push([snap.ts, a.price]);
      if (h.length > 600) h.splice(0, h.length - 600);
    }
    drawSpark(sym);

    for (const tf of TF_ORDER) renderTf(sym, tf, a.tfs ? a.tfs[tf] : null);
    renderNews(sym, snap);
  }

  renderBgn(snap.bgn);
  renderWeights(snap.weights, snap.weight_labels || snap.formula_names);
  renderFormulaMonitor(snap);
  renderBrainPanel(snap);
  renderEmotionPanel(snap);
  renderRiskPanel(snap);
}

function renderTf(sym, tf, cyc) {
  const dirEl = document.getElementById(`dir-${sym}-${tf}`);
  const confB = document.getElementById(`confb-${sym}-${tf}`);
  const confN = document.getElementById(`confn-${sym}-${tf}`);
  const tgts = document.getElementById(`tgts-${sym}-${tf}`);
  const risk = document.getElementById(`risk-${sym}-${tf}`);
  const nt = document.getElementById(`nt-${sym}-${tf}`);
  const pr = document.getElementById(`pr-${sym}-${tf}`);
  const warn = document.getElementById(`warn-${sym}-${tf}`);
  const cnt = document.getElementById(`cnt-${sym}-${tf}`);
  if (!cyc) {
    dirEl.textContent = "◆ awaiting"; dirEl.className = "tf-dir flat";
    return;
  }

  dirEl.textContent = cyc.issued
    ? (cyc.direction > 0 ? "▲ BULLISH" : "▼ BEARISH")
    : "◆ NO SIGNAL";
  dirEl.className = "tf-dir " + (cyc.issued ? (cyc.direction > 0 ? "up" : "down") : "flat");

  const pct = Math.round(cyc.confidence * 100);
  confB.style.width = pct + "%";
  confB.className = "conf-fill " + (cyc.confidence > 0.8 ? "strong" : cyc.confidence < 0.5 ? "weak" : "");
  confN.textContent = pct + "%";

  // targets
  tgts.innerHTML = (cyc.targets || []).map(t => {
    const active = cyc.active_k === t.k && t.status === "locked";
    const cls = active ? "active" : t.status;
    const st = t.status === "pending" ? "CALC ON HIT" : t.status.toUpperCase();
    const lvl = t.level != null ? fmtPrice(t.level) : "—";
    const arrow = t.direction > 0 ? "▲" : t.direction < 0 ? "▼" : "·";
    return `<div class="tgt ${cls}">
      <span class="k">Tp${t.k}</span><span>${t.minutes}m</span>
      <span>${lvl}</span><span>${t.status === "locked" || t.status === "hit" ? arrow : ""}</span>
      <span class="st">${active ? "▸ " : ""}${st}</span></div>`;
  }).join("");

  risk.innerHTML = `
    <span class="tp">TP ${cyc.tp != null ? fmtPrice(cyc.tp) : "—"}</span>
    <span class="sl">SL ${cyc.sl != null ? fmtPrice(cyc.sl) : "—"}</span>
    <span class="ratio-badge">${cyc.ratio ? (cyc.ratio === 3 ? "3:1" : cyc.ratio === 1.5 ? "3:2" : "1:1") : "—"}</span>`;

  nt.classList.toggle("hidden", !!cyc.issued);
  pr.classList.toggle("hidden", !cyc.pending_reset);
  warn.classList.toggle("hidden", !cyc.warning);
  cnt.textContent = fmtMMSS(cyc.countdown_s != null && cyc.cycle_end
    ? (cyc.cycle_end - (state.snap ? state.snap.ts : Date.now() / 1000))
    : cyc.countdown_s);
  cnt.classList.toggle("urgent", cyc.cycle_end && (cyc.cycle_end - Date.now() / 1000) < 60);
}

function renderNews(sym, snap) {
  const el = document.getElementById(`news-${sym}`);
  const systems = snap.systems && snap.systems[`${sym}|tf1`];
  if (!systems || !systems.news) { el.innerHTML = ""; return; }
  const items = systems.news.items || [];
  el.innerHTML = items.length
    ? items.map(it => `<span class="news-chip ${it.direction > 0 ? "bull" : "bear"}">
        ${it.direction > 0 ? "▲" : "▼"} ${(it.impact * 100).toFixed(0)}% · ${it.headline}</span>`).join("")
    : `<span class="news-chip">no active headlines</span>`;
}

/* ---------- sparkline with TP/SL/target markers ---------- */
function drawSpark(sym) {
  const cv = document.getElementById(`spark-${sym}`);
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const hist = state.history[sym];
  if (hist.length < 2) return;
  const prices = hist.map(x => x[1]);
  let lo = Math.min(...prices), hi = Math.max(...prices);

  // include active target / tp / sl levels of all TFs for context
  const snap = state.snap;
  const markers = [];
  if (snap && snap.assets[sym]) {
    for (const tf of TF_ORDER) {
      const c = snap.assets[sym].tfs && snap.assets[sym].tfs[tf];
      if (!c || !c.issued) continue;
      if (c.sl != null) markers.push([c.sl, "#ff5d73", 0.7]);
      if (c.tp != null) markers.push([c.tp, "#2fd48f", 0.7]);
      const ak = c.active_k || 0;
      const at = c.targets && c.targets[ak - 1];
      if (at && at.level != null && at.status === "locked")
        markers.push([at.level, "#ffb02e", 1]);
    }
  }
  for (const [lv] of markers) { lo = Math.min(lo, lv); hi = Math.max(hi, lv); }
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;

  const X = i => (i / (hist.length - 1)) * w;
  const Y = p => h - ((p - lo) / (hi - lo)) * h;

  // grid
  ctx.strokeStyle = "rgba(31,43,69,0.7)"; ctx.lineWidth = 1;
  for (let g = 1; g < 4; g++) {
    ctx.beginPath(); ctx.moveTo(0, (h * g) / 4); ctx.lineTo(w, (h * g) / 4); ctx.stroke();
  }
  // markers
  ctx.setLineDash([4, 4]);
  for (const [lv, col] of markers) {
    ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, Y(lv)); ctx.lineTo(w, Y(lv)); ctx.stroke();
  }
  ctx.setLineDash([]);
  // line
  const up = prices[prices.length - 1] >= prices[0];
  ctx.beginPath();
  hist.forEach(([ts, p], i) => { i ? ctx.lineTo(X(i), Y(p)) : ctx.moveTo(X(i), Y(p)); });
  ctx.strokeStyle = up ? "#2fd48f" : "#ff5d73";
  ctx.lineWidth = 1.6;
  ctx.stroke();
  // glow head
  const lastX = X(hist.length - 1), lastY = Y(prices[prices.length - 1]);
  ctx.beginPath(); ctx.arc(lastX, lastY, 2.6, 0, Math.PI * 2);
  ctx.fillStyle = up ? "#2fd48f" : "#ff5d73"; ctx.fill();
}

/* ---------- Bgn matrix / weights / formula monitor ---------- */
function renderBgn(matrix) {
  const table = document.getElementById("bgn-table");
  if (!matrix || !Object.keys(matrix).length) return;
  const tfs = TF_ORDER;
  table.tHead.innerHTML = `<tr><th></th>${tfs.map(t => `<th>${t.toUpperCase()}</th>`).join("")}</tr>`;
  table.tBodies[0].innerHTML = tfs.map(ti =>
    `<tr><th>${ti.toUpperCase()}</th>${tfs.map(tj => {
      const v = matrix[ti] && matrix[ti][tj];
      const a = v == null ? 0 : Math.min(1, Math.max(0, v));
      return `<td style="background:rgba(91,140,255,${(a * 0.35).toFixed(3)})">${v != null ? v.toFixed(3) : "—"}</td>`;
    }).join("")}</tr>`).join("");
}

function renderWeights(weights, names) {
  const el = document.getElementById("weights-list");
  if (!weights) return;
  const maxW = Math.max(...Object.values(weights), 0.15);
  el.innerHTML = Object.keys(weights).sort((a, b) => +a - +b).map(f => {
    const w = weights[f];
    return `<div class="wrow">
      <span class="wname">F${f} ${names ? names[f] : ""}</span>
      <span class="wbar"><span class="wfill" style="width:${(w / maxW * 100).toFixed(1)}%"></span></span>
      <span class="wval">${w.toFixed(3)}</span></div>`;
  }).join("");
}

function renderFormulaMonitor(snap) {
  const el = document.getElementById("formula-monitor");
  for (const sym of SYMBOLS) {
    let box = document.getElementById(`fm-${sym}`);
    if (!box) {
      box = document.createElement("div");
      box.className = "fm-asset";
      box.id = `fm-${sym}`;
      box.innerHTML = `<div class="fm-head">
          <span>${sym}</span>
          <span class="fm-tf-select">${TF_ORDER.map(tf =>
            `<button data-sym="${sym}" data-tf="${tf}" class="${tf === "tf1" ? "on" : ""}">${tf.toUpperCase()}</button>`).join("")}</span>
        </div><div class="fm-grid" id="fmgrid-${sym}"></div>`;
      el.appendChild(box);
      box.querySelectorAll(".fm-tf-select button").forEach(b => {
        b.addEventListener("click", () => {
          state.fmTf[b.dataset.sym] = b.dataset.tf;
          box.querySelectorAll(".fm-tf-select button").forEach(x =>
            x.classList.toggle("on", x.dataset.tf === b.dataset.tf));
          if (state.snap) renderFormulaMonitor(state.snap);
        });
      });
    }
    const tf = state.fmTf[sym];
    const data = snap.formula_values && snap.formula_values[`${sym}|${tf}`];
    const grid = document.getElementById(`fmgrid-${sym}`);
    if (!data) { grid.innerHTML = `<div class="fm-item">awaiting first GRC…</div>`; continue; }
    grid.innerHTML = Object.keys(data.adj).sort((a, b) => +a - +b).map(f => {
      const v = data.adj[f];
      const wpc = Math.min(50, Math.abs(v) * 50);
      const cls = v >= 0 ? "pos" : "neg";
      const style = v >= 0 ? `width:${wpc}%` : `width:${wpc}%`;
      return `<div class="fm-item" title="raw=${data.raw[f]}">
        <span class="fid">F${f}</span><span class="fname">${data.names[f]}</span>
        <span class="fbar"><span class="zero"></span><span class="fval ${cls}" style="${style}"></span></span>
        <span class="fnum">${v >= 0 ? "+" : ""}${v.toFixed(3)}</span></div>`;
    }).join("");
  }
}

/* ---------- Part-2 panels: brains / emotions / risk ---------- */
function detailSelectorHTML() {
  const { sym, tf } = state.detail;
  return `<span class="sel-group">${SYMBOLS.map(s =>
    `<button data-kind="sym" data-v="${s}" class="${s === sym ? "on" : ""}">${s}</button>`).join("")}</span>
    <span class="sel-group">${TF_ORDER.map(t =>
    `<button data-kind="tf" data-v="${t}" class="${t === tf ? "on" : ""}">${TF_LABEL[t]}</button>`).join("")}</span>`;
}

function wireSelector(id, rerender) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = detailSelectorHTML();
  el.querySelectorAll("button").forEach(b => {
    b.addEventListener("click", () => {
      state.detail[b.dataset.kind] = b.dataset.v;
      rerender(state.snap);
    });
  });
}

function renderBrainPanel(snap) {
  if (!snap) return;
  wireSelector("brain-selector", renderBrainPanel);
  const el = document.getElementById("brain-panel");
  const sys = snap.systems && snap.systems[`${state.detail.sym}|${state.detail.tf}`];
  const info = snap.brains || {};
  const conn = info.connectome || {};
  const head = `<div class="conn-line">
    <span>connectome: <b>${conn.source || "—"}</b></span>
    <span>${conn.units_per_brain ?? "—"} units/brain</span>
    <span>${conn.edges != null ? conn.edges.toLocaleString() : "—"} edges</span>
    <span>${conn.layout || ""}</span>
    <span>instances: ${(info.instances || []).map(i => `${i.id}${i.ready ? " ✓" : " …"}`).join(" · ")}</span>
  </div>`;
  if (!sys || !sys.brain) { el.innerHTML = head + `<div class="fm-item">awaiting first GRC…</div>`; return; }
  const br = sys.brain;
  const polBadge = br.polarity > 0 ? `<span class="pol bull">▲ BULLISH</span>`
    : br.polarity < 0 ? `<span class="pol bear">▼ BEARISH</span>` : `<span class="pol flat">◆ NEUTRAL</span>`;
  const rows = (br.instances || []).map(i => {
    const hPct = Math.round((i.hormone_state || 0) * 100);
    const sPct = Math.round(((i.h_signal || 0) + 1) / 2 * 100);
    return `<tr>
      <td><b>${i.id}</b><span class="role">${i.role || ""}</span></td>
      <td>H ${hPct}%<span class="mini-bar"><span style="width:${hPct}%"></span></span></td>
      <td class="${(i.h_signal || 0) >= 0 ? "pos" : "neg"}">${(i.h_signal || 0) >= 0 ? "+" : ""}${(i.h_signal || 0).toFixed(4)}</td>
      <td>EQ ${(i.experience || 0).toFixed(3)}</td>
      <td>pat ${(i.pattern || 0) >= 0 ? "+" : ""}${(i.pattern || 0).toFixed(3)}</td>
      <td>${i.dominant || "—"} ${(i.p_dominant || 0).toFixed(2)}</td>
      <td><span class="mini-bar wide"><span style="width:${sPct}%"></span></span></td>
    </tr>`;
  }).join("");
  const hormones = Object.entries(br.hormones_avg || {}).map(([k, v]) =>
    `<span class="horm">I_${k} <b>${(+v).toFixed(3)}</b></span>`).join("");
  el.innerHTML = `${head}
    <div class="consensus-line">
      <b>${state.detail.sym} ${state.detail.tf.toUpperCase()}</b>
      consensus ${br.consensus >= 0 ? "+" : ""}${(+br.consensus).toFixed(4)} ${polBadge}
      strength ${(br.strength || 0).toFixed(3)} · agreement ${((br.agreement || 0) * 100).toFixed(0)}%
      · EQ(B2) ${(br.experience || 0).toFixed(3)} · P(dom) ${(br.p_dominant || 0).toFixed(2)}
    </div>
    <div class="horm-row">${hormones}</div>
    <table class="brain-table"><thead><tr>
      <th>Instance / role</th><th>MBON H</th><th>H_signal</th><th>Experience</th><th>Pattern</th><th>Dominant state</th><th>readout</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

function renderEmotionPanel(snap) {
  if (!snap) return;
  wireSelector("emotion-selector", renderEmotionPanel);
  const el = document.getElementById("emotion-panel");
  const sys = snap.systems && snap.systems[`${state.detail.sym}|${state.detail.tf}`];
  if (!sys || !sys.emotion) { el.innerHTML = `<div class="fm-item">awaiting first GRC…</div>`; return; }
  const em = sys.emotion;
  const chips = [];
  for (let eid = 1; eid <= 22; eid++) {
    const meta = EMOTION_META[eid];
    const val = eid === 22 ? (em.cerc ?? em.aggregate) : (em.emotions ? em.emotions[eid] : null);
    if (val == null) continue;
    const dir = eid === 22 ? Math.sign(val) : (em.directions ? em.directions[eid] : 0);
    const arrow = dir > 0 ? "▲" : dir < 0 ? "▼" : "·";
    const dcls = dir > 0 ? "up" : dir < 0 ? "down" : "flat";
    const v = Math.max(0, Math.min(1, Math.abs(eid === 22 ? val : val)));
    chips.push(`<div class="emo-chip ${eid === 22 ? "cerc" : ""}" title="E${eid} ${meta[1]}">
      <span class="eid">E${eid}</span><span class="ename">${meta[0]}</span>
      <span class="ebar"><span style="width:${(v * 100).toFixed(0)}%"></span></span>
      <span class="eval">${(+val).toFixed(3)}</span>
      <span class="edir ${dcls}">${arrow}</span></div>`);
  }
  const weights = (em.weights || []).map((w, i) => {
    const meta = EMOTION_META[i + 1];
    return `<div class="wrow slim">
      <span class="wname">E${i + 1} ${meta[0]}</span>
      <span class="wbar"><span class="wfill" style="width:${Math.min(100, w / 0.15 * 100).toFixed(1)}%"></span></span>
      <span class="wval">${w.toFixed(3)}</span></div>`;
  }).join("");
  const cerc = +(em.cerc ?? em.aggregate ?? 0);
  el.innerHTML = `
    <div class="consensus-line">
      <b>${state.detail.sym} ${state.detail.tf.toUpperCase()}</b>
      E22 CERC <b class="${cerc >= 0 ? "pos" : "neg"}">${cerc >= 0 ? "+" : ""}${cerc.toFixed(4)}</b>
      → CCI component 98 · aggregate ${(+em.aggregate).toFixed(4)}
      · RTP ${(em.rtp ?? 0).toFixed(3)} · TEDR ${(em.tedr ?? 0).toFixed(3)}
      <span class="pol ${em.polarity > 0 ? "bull" : em.polarity < 0 ? "bear" : "flat"}">
        ${em.polarity > 0 ? "▲" : em.polarity < 0 ? "▼" : "◆"}</span>
    </div>
    <div class="emo-grid">${chips.join("")}</div>
    <div class="emo-weights"><div class="sub-h">CERC weights w<sub>e</sub> (init 1/21 · η = 0.02 · clamp [0.005, 0.15])</div>${weights}</div>`;
}

function renderRiskPanel(snap) {
  if (!snap) return;
  wireSelector("risk-selector", renderRiskPanel);
  const el = document.getElementById("risk-panel");
  if (!snap.systems) { el.innerHTML = `<div class="fm-item">awaiting first GRC…</div>`; return; }
  // per-(asset, tf) matrix of F28 ratio bands + conflict flags
  const matrix = `<table class="risk-matrix"><thead><tr><th></th>${TF_ORDER.map(t =>
    `<th>${TF_LABEL[t]}</th>`).join("")}</tr></thead><tbody>${SYMBOLS.map(s =>
    `<tr><th>${s}</th>${TF_ORDER.map(t => {
      const x = snap.systems[`${s}|${t}`];
      if (!x) return `<td>—</td>`;
      const band = x.arc_ratio ? (x.arc_ratio === 3 ? "3:1" : x.arc_ratio === 1.5 ? "3:2" : "1:1") : "NO TRD";
      const cls = x.arc_ratio === 3 ? "r31" : x.arc_ratio === 1.5 ? "r32" : x.arc_ratio === 1 ? "r11" : "rnt";
      const warn = x.conflicting ? `<span class="conflict-flag" title="F30 PCV conflicting signals">⚠</span>` : "";
      return `<td class="band ${cls}" title="ARC ${(+x.arc).toFixed(3)} · PCV ${(+x.pcv).toFixed(3)}">${band}${warn}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
  const x = snap.systems[`${state.detail.sym}|${state.detail.tf}`];
  if (!x) { el.innerHTML = matrix; return; }
  const arc = +(x.arc ?? 0);
  const band = x.arc_ratio ? (x.arc_ratio === 3 ? "3:1" : x.arc_ratio === 1.5 ? "3:2" : "1:1") : "NO TRADE (conf < 0.35)";
  const pcv = +(x.pcv ?? 0);
  const conflict = x.conflicting || pcv < 0;
  const sig = x.sigma || {};
  const detail = `<div class="risk-detail">
    <div class="rd-cell"><label>F28 ARC</label><b>${arc.toFixed(3)}</b><span>→ ${band}</span></div>
    <div class="rd-cell"><label>σ current</label><b>${(+sig.cur || 0).toExponential(2)}</b><span>σ base ${(+sig.base || 0).toExponential(2)}</span></div>
    <div class="rd-cell"><label>F30 PCV</label><b class="${pcv < 0 ? "neg" : "pos"}">${pcv >= 0 ? "+" : ""}${pcv.toFixed(4)}</b>
      <span>${conflict ? "⚠ CONFLICTING SIGNALS" : "aligned with CCI"}</span></div>
    <div class="rd-cell"><label>Confidence</label><b>${((x.confidence ?? 0) * 100).toFixed(1)}%</b>
      <span>base ${((x.base_confidence ?? 0) * 100).toFixed(1)}% · DRHE ${(+x.drhe ?? 0).toFixed(3)}</span></div>
    <div class="rd-cell"><label>ATR</label><b>${(+x.atr ?? 0).toFixed(4)}</b><span>win streak ${(+x.win_ratio ?? 0).toFixed(2)}</span></div>
  </div>`;
  const confLine = conflict
    ? `<div class="conflict-banner">⚠ CONFLICTING SIGNALS — F30 PCV opposes the CCI · confidence −|PCV| × 0.3</div>`
    : "";
  el.innerHTML = `${matrix}<div class="sub-h">${state.detail.sym} ${state.detail.tf.toUpperCase()} detail</div>${detail}${confLine}`;
}

/* ======================= connection ======================= */
function setConn(status, label) {
  const el = document.getElementById("conn-status");
  el.className = "conn " + status;
  el.textContent = label;
}

function handleMessage(msg) {
  switch (msg.type) {
    case "snapshot":
      renderSnapshot(msg);
      break;
    case "phase":
      renderPhase(msg);
      break;
    case "tick": {
      if (!state.snap) break;
      state.snap.ts = msg.ts;
      for (const sym of SYMBOLS) {
        const p = msg.prices[sym];
        if (p > 0) {
          if (state.snap.assets[sym]) state.snap.assets[sym].price = p;
          const el = document.getElementById(`price-${sym}`);
          if (el) el.textContent = fmtPrice(p);
          const h = state.history[sym];
          h.push([msg.ts, p]);
          if (h.length > 600) h.splice(0, h.length - 600);
        }
      }
      for (const sym of SYMBOLS) drawSpark(sym);
      break;
    }
    case "sl_event":
      if (msg.kind === "close") {
        showBanner("close", "CLOSE POSITION IMMEDIATELY",
          `${msg.symbol} ${msg.tf.toUpperCase()} — prediction contradicted by full recalculation`);
        alertSound("close");
        haptic();
      } else {
        showBanner("safe", "SAFE — BE CAUTIOUS",
          `${msg.symbol} ${msg.tf.toUpperCase()} — SL tightened ${msg.sl_tighten_pct}% · confidence −0.15`);
        alertSound("safe");
      }
      break;
    case "event":
      if (msg.kind === "target_locked") {
        /* silent — panel updates on next snapshot tick */
      }
      break;
  }
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => { setConn("connected", "● realtime connected"); state.connected = true; };
  ws.onmessage = ev => {
    try { handleMessage(JSON.parse(ev.data)); } catch (e) { console.error(e); }
  };
  ws.onclose = () => {
    state.connected = false;
    setConn("disconnected", "◌ reconnecting…");
    setTimeout(connectWS, 2000);
    pollFallback();
  };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}

/* REST fallback if WS is blocked by the proxy */
let pollTimer = null;
async function pollFallback() {
  if (state.connected) { clearTimeout(pollTimer); return; }
  try {
    const r = await fetch("/api/snapshot");
    if (r.ok) handleMessage(await r.json());
  } catch (e) {}
  pollTimer = setTimeout(pollFallback, 1500);
}

/* ======================= 1s local countdown ticker ======================= */
setInterval(() => {
  const snap = state.snap;
  if (!snap) return;
  // GRC countdown
  const now = Date.now() / 1000;
  const toGrc = Math.max(0, snap.grc.next - now);
  const el = document.getElementById("grc-countdown");
  el.textContent = fmtMMSS(toGrc);
  el.style.color = toGrc < 30 ? "#ff5d73" : "#ffb02e";
  // per-TF countdowns
  for (const sym of SYMBOLS) {
    const a = snap.assets[sym];
    if (!a) continue;
    for (const tf of TF_ORDER) {
      const c = a.tfs && a.tfs[tf];
      if (!c) continue;
      const remain = Math.max(0, (c.cycle_end || 0) - now);
      const cnt = document.getElementById(`cnt-${sym}-${tf}`);
      if (cnt) {
        cnt.textContent = fmtMMSS(remain);
        cnt.classList.toggle("urgent", remain < 60 && remain > 0);
      }
    }
  }
}, 1000);

/* ======================= boot ======================= */
buildGrid();
connectWS();
pollFallback();
window.addEventListener("resize", () => SYMBOLS.forEach(drawSpark));
