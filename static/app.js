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
  onSnapshotExtras(snap);
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
      onDetailChange();
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

/* ======================= PART 3: tabs / chart / gauges / panels =======================
 * Data contracts:
 *   GET /api/formula-history?symbol&tf  -> core(31) + emotion(22) + physics(22)
 *      sparklines, accuracy rings, names, CCI weights, physics alpha
 *   GET /api/candles?symbol&n           -> OHLC bars (+ live partial minute)
 *   GET /api/news/{symbol}             -> signed aggregate, per-TF impacts, items
 *   GET /api/brain-activations         -> 4 x u8 b64 (drives the WebGL flash)
 *   snapshot.systems[|tf]              -> physics values/alpha/couplings/p1,
 *      news, hedge, corelation (50-GRC network), brain consensus, emotion
 */
const NON_DIR_P = new Set([2, 6, 9, 19, 21]);   // P22 non-directional inputs
const stateExtras = {
  tab: "dash",
  fh: null, fhFor: "", fhAt: 0,
  news: null, newsFor: "", newsAt: 0,
  bars: [], barsFor: "", barsAt: 0,
  actAt: 0, lastGrc: -1,
  corelTipNode: null,
};

function activeTab() { return stateExtras.tab; }

function switchTab(name) {
  if (!["dash", "brains", "formulas", "corel", "news"].includes(name)) name = "dash";
  stateExtras.tab = name;
  document.querySelectorAll(".tab-btn").forEach(b =>
    b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tabpage").forEach(p =>
    p.classList.toggle("on", p.id === "tab-" + name));
  if (window.brain3D) window.brain3D.setVisible(name === "brains");
  if (name === "formulas") ensureFH(true);
  if (name === "news") ensureNews(true);
  if (name === "corel") renderCorel();
  if (name === "dash") { drawChart(); renderGauges(state.snap); }
  if (name === "brains") refreshActivations(true);
  try { history.replaceState(null, "", "#" + name); } catch (e) {}
}

function wireTabs() {
  document.querySelectorAll(".tab-btn").forEach(b =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
  const h = (location.hash || "").replace("#", "");
  if (h) switchTab(h);
}

/* selectors that mirror state.detail (shared with brain/emotion/risk panels) */
function wireExtraSelectors() {
  wireSelector("chart-selector", () => { renderGauges(state.snap); drawChart(); });
  wireSelector("gauge-selector", renderGauges);
  wireSelector("panels-selector", () => { stateExtras.fhFor = ""; ensureFH(true); renderPanels(state.snap); });
  wireSelector("corel-selector", renderCorel);
  wireSelector("news-selector", () => { stateExtras.newsFor = ""; ensureNews(true); });
  wireSelector("brain3d-selector", () => {
    refreshActivations(true);
    renderBrain3dSide(state.snap);
  });
}

/* called by renderSnapshot + detail changes */
function onSnapshotExtras(snap) {
  if (!snap) return;
  const grcChanged = snap.grc.count !== stateExtras.lastGrc;
  if (grcChanged) stateExtras.lastGrc = snap.grc.count;
  renderGauges(snap);
  if (stateExtras.tab === "dash") drawChart();
  if (stateExtras.tab === "formulas") { if (grcChanged) stateExtras.fhAt = 0; ensureFH(); renderPanels(snap); }
  if (stateExtras.tab === "news") { if (grcChanged) stateExtras.newsAt = 0; ensureNews(); renderNewsTab(snap); renderHF(snap); }
  if (stateExtras.tab === "corel") renderCorel(snap);
  if (stateExtras.tab === "brains") renderBrain3dSide(snap);
  if (grcChanged) refreshActivations();
  if (stateExtras.tab !== "formulas") renderPanels.deferred = snap;
}

function onDetailChange() {
  stateExtras.fhFor = "";
  stateExtras.newsFor = "";
  if (stateExtras.tab === "formulas") ensureFH(true);
  if (stateExtras.tab === "news") ensureNews(true);
  if (stateExtras.tab === "corel") renderCorel();
  refreshActivations(true);
  stateExtras.barsAt = 0;
  drawChart();
  renderGauges(state.snap);
}

/* ---------------- gauges ---------------- */
const GAUGE_DEFS = [
  { key: "confidence", label: "CONFIDENCE", lo: 0, hi: 1, fmt: v => (v * 100).toFixed(1) + "%", warn: v => v > 0.35 },
  { key: "base_confidence", label: "BASE CCI", lo: 0, hi: 1, fmt: v => (v * 100).toFixed(1) + "%" },
  { key: "ufp", label: "PHYSICS UFP", lo: -1, hi: 1, fmt: v => v.toFixed(3) },
  { key: "tme", label: "P9 · TME", lo: 0, hi: 1, fmt: v => v.toFixed(3) },
  { key: "wfcp", label: "P2 · WFCP", lo: 0, hi: 1, fmt: v => v.toFixed(3) },
  { key: "drpt", label: "P6 · DRPT", lo: 0, hi: 1, fmt: v => v.toFixed(3) },
  { key: "gspi", label: "P19 · GSPI", lo: 0, hi: 1, fmt: v => v.toFixed(3) },
  { key: "news", label: "NEWS (signed)", lo: -1, hi: 1, fmt: v => v.toFixed(3) },
  { key: "cerc", label: "E22 · CERC", lo: -1, hi: 1, fmt: v => v.toFixed(3) },
];

function gaugeValue(sys) {
  if (!sys) return {};
  return {
    confidence: sys.confidence,
    base_confidence: sys.base_confidence,
    ufp: sys.physics && sys.physics.ufp,
    tme: sys.physics && sys.physics.tme,
    wfcp: sys.physics && sys.physics.wfcp,
    drpt: sys.physics && sys.physics.drpt,
    gspi: sys.physics && sys.physics.gspi,
    news: sys.news && sys.news.value,
    cerc: sys.emotion && sys.emotion.cerc,
  };
}

function drawGauge(cv, v, def) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 150, h = cv.clientHeight || 92;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h - 14, R = Math.min(w / 2 - 10, h - 26);
  const a0 = Math.PI * 0.9, a1 = Math.PI * 2.1;
  /* track */
  ctx.beginPath(); ctx.arc(cx, cy, R, a0, a1);
  ctx.lineWidth = 10; ctx.lineCap = "round";
  ctx.strokeStyle = "#1c2a45"; ctx.stroke();
  const t = v == null ? null : Math.max(0, Math.min(1, (v - def.lo) / (def.hi - def.lo)));
  if (t != null) {
    const neg = def.lo < 0 && v < 0;
    ctx.beginPath();
    ctx.arc(cx, cy, R, a0, a0 + (a1 - a0) * t);
    ctx.lineWidth = 10;
    ctx.strokeStyle = neg ? "#ff5d73" : (def.warn && def.warn(v) ? "#2fd48f" : "#ffb02e");
    ctx.stroke();
    /* needle */
    const an = a0 + (a1 - a0) * t;
    ctx.beginPath();
    ctx.moveTo(cx - Math.cos(an) * 6, cy - Math.sin(an) * 6);
    ctx.lineTo(cx + Math.cos(an) * (R - 14), cy + Math.sin(an) * (R - 14));
    ctx.strokeStyle = "#e8eefc"; ctx.lineWidth = 1.6; ctx.stroke();
  }
  ctx.fillStyle = "#8ea0c4"; ctx.font = "10px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(def.label, cx, h - 2);
  ctx.fillStyle = "#e8eefc"; ctx.font = "600 13px system-ui";
  ctx.fillText(v == null ? "—" : def.fmt(v), cx, cy - R * 0.34);
}

function renderGauges(snap) {
  const row = document.getElementById("gauge-row");
  if (!row) return;
  const key = `${state.detail.sym}|${state.detail.tf}`;
  const sys = snap && snap.systems && snap.systems[key];
  const vals = gaugeValue(sys);
  if (!row.children.length) {
    row.innerHTML = GAUGE_DEFS.map((d, i) =>
      `<div class="gauge"><canvas id="gauge-${i}"></canvas></div>`).join("");
  }
  GAUGE_DEFS.forEach((d, i) => {
    const cv = document.getElementById(`gauge-${i}`);
    if (cv) drawGauge(cv, vals[d.key], d);
  });
}

/* ---------------- candle chart + TP/SL + P1 overlay ---------------- */
async function ensureBars(force) {
  const sym = state.detail.sym;
  const fresh = Date.now() - stateExtras.barsAt < 12000;
  if (!force && stateExtras.barsFor === sym && fresh) return;
  try {
    const r = await fetch(`/api/candles?symbol=${sym}&n=240`);
    if (!r.ok) return;
    const j = await r.json();
    stateExtras.bars = j.bars || [];
    stateExtras.barsFor = sym;
    stateExtras.barsAt = Date.now();
    if (stateExtras.tab === "dash" && activeTab() === "dash") drawChart();
  } catch (e) { /* retry next tick */ }
}

function drawChart() {
  const cv = document.getElementById("chart-canvas");
  if (!cv || stateExtras.tab !== "dash") return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const bars = stateExtras.bars;
  const sym = state.detail.sym, tf = state.detail.tf;
  const gutter = 132;                       // right gutter for the P1 ladder
  const pw = w - gutter;
  if (!bars || bars.length < 3) {
    ctx.fillStyle = "#8ea0c4"; ctx.font = "12px system-ui";
    ctx.fillText("loading bars…", 14, 24);
    return;
  }

  /* levels: TP/SL/targets of this pair + P1 quantum levels */
  const cyc = state.snap && state.snap.assets[sym] && state.snap.assets[sym].tfs[tf];
  const lgTp = document.querySelector("#chart-legend .lg.tp");
  const lgSl = document.querySelector("#chart-legend .lg.sl");
  const lgTg = document.querySelector("#chart-legend .lg.tgt");
  if (lgTp) lgTp.textContent = cyc && cyc.tp != null ? "TP " + fmtPrice(cyc.tp) : "— TP";
  if (lgSl) lgSl.textContent = cyc && cyc.sl != null ? "SL " + fmtPrice(cyc.sl) : "— SL";
  let activeT = null;
  if (cyc && cyc.active_k && cyc.targets) {
    const t = (cyc.targets || []).find(x => x.k === cyc.active_k);
    if (t && t.level != null && t.status === "locked") activeT = t;
  }
  if (lgTg) lgTg.textContent = activeT ? `active target Tp${activeT.k} ${fmtPrice(activeT.level)}` : "— active target";
  const levels = [];
  if (cyc && cyc.issued) {
    if (cyc.sl != null) levels.push([cyc.sl, "#ff5d73", "SL", 2]);
    if (cyc.tp != null) levels.push([cyc.tp, "#2fd48f", "TP", 2]);
    (cyc.targets || []).forEach(t => {
      if (t.level != null && (t.status === "locked" || t.status === "hit"))
        levels.push([t.level, "#ffb02e", `Tp${t.k}`, 1.4]);
    });
  }
  const sys = state.snap && state.snap.systems && state.snap.systems[`${sym}|${tf}`];
  const p1 = sys && sys.physics && sys.physics.p1;
  const ladder = [];
  if (p1 && p1.levels && p1.probs) {
    for (let i = 0; i < p1.levels.length; i++)
      ladder.push([p1.levels[i], p1.probs[i]]);
  }

  let lo = Math.min(...bars.map(b => b.l)), hi = Math.max(...bars.map(b => b.h));
  for (const [lv] of levels) { lo = Math.min(lo, lv); hi = Math.max(hi, lv); }
  if (ladder.length) {
    lo = Math.min(lo, ladder[0][0]);
    hi = Math.max(hi, ladder[ladder.length - 1][0]);
  }
  const pad = (hi - lo) * 0.06 || 1;
  lo -= pad; hi += pad;
  const Y = p => h - 18 - ((p - lo) / (hi - lo)) * (h - 30);
  const step = pw / bars.length;
  const bw = Math.max(1.5, step * 0.62);

  /* grid + price axis */
  ctx.strokeStyle = "rgba(31,43,69,0.7)"; ctx.fillStyle = "#6b7ba0";
  ctx.font = "10px system-ui"; ctx.textAlign = "left";
  for (let g = 0; g <= 4; g++) {
    const y = 6 + ((h - 24) * g) / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(pw, y); ctx.stroke();
    const price = hi - ((hi - lo) * g) / 4;
    ctx.fillText(fmtPrice(price), pw + 4, y + 3);
  }

  /* candles */
  bars.forEach((b, i) => {
    const x = i * step + step / 2;
    const up = b.c >= b.o;
    ctx.strokeStyle = ctx.fillStyle = up ? "#2fd48f" : "#ff5d73";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, Y(b.h)); ctx.lineTo(x, Y(b.l)); ctx.stroke();
    const yO = Y(b.o), yC = Y(b.c);
    const top = Math.min(yO, yC), bh = Math.max(1, Math.abs(yC - yO));
    ctx.fillRect(x - bw / 2, top, bw, bh);
  });

  /* TP / SL / target layers */
  ctx.setLineDash([6, 5]);
  for (const [lv, col, tag, lw] of levels) {
    ctx.strokeStyle = col; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(0, Y(lv)); ctx.lineTo(pw, Y(lv)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.font = "600 10px system-ui";
    ctx.fillText(tag + " " + fmtPrice(lv), 6, Y(lv) - 3);
    ctx.setLineDash([6, 5]);
  }
  ctx.setLineDash([]);

  /* live price dot */
  const last = bars[bars.length - 1];
  ctx.beginPath(); ctx.arc(bars.length * step - step / 2, Y(last.c), 3, 0, Math.PI * 2);
  ctx.fillStyle = "#e8eefc"; ctx.fill();

  /* -------- P1 probability overlay (right gutter ladder) -------- */
  if (ladder.length) {
    const gx = pw + 56;
    ctx.fillStyle = "#5f719a"; ctx.font = "600 9px system-ui"; ctx.textAlign = "left";
    ctx.fillText("P1 · max|c|²", gx, 12);
    const maxP = Math.max(...ladder.map(l => l[1]), 1e-9);
    ladder.forEach(([lvl, prob]) => {
      const y = Y(lvl);
      if (y < 4 || y > h - 8) return;
      const bwid = 64 * (prob / maxP);
      ctx.fillStyle = "rgba(47,212,143,0.75)";
      ctx.fillRect(gx, y - 3, Math.max(1, bwid), 6);
      ctx.fillStyle = "#9fb0d4"; ctx.font = "9px system-ui";
      if (prob >= maxP * 0.35) {
        ctx.fillText(prob.toFixed(3), gx + bwid + 3, y + 3);
      }
    });
    /* headline: highest-probability level */
    let best = ladder[0];
    for (const l of ladder) if (l[1] > best[1]) best = l;
    ctx.fillStyle = "#ffd166"; ctx.font = "600 10px system-ui";
    ctx.fillText("peak " + fmtPrice(best[0]), gx, h - 4);
  }
}

/* ---------------- 75 formula panels ---------------- */
async function ensureFH(force) {
  const { sym, tf } = state.detail;
  const tag = `${sym}|${tf}`;
  const fresh = Date.now() - stateExtras.fhAt < 30000;
  if (!force && stateExtras.fhFor === tag && fresh && stateExtras.fh) return;
  try {
    const r = await fetch(`/api/formula-history?symbol=${sym}&tf=${tf}`);
    if (!r.ok) return;
    const j = await r.json();
    stateExtras.fh = j;
    stateExtras.fhFor = tag;
    stateExtras.fhAt = Date.now();
    renderPanels(state.snap);
  } catch (e) { /* retry */ }
}

function drawMini(cv, arr, color) {
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 120, h = cv.clientHeight || 26;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!arr || arr.length < 2) {
    ctx.fillStyle = "#43506e"; ctx.font = "9px system-ui";
    ctx.fillText("awaiting…", 4, h / 2 + 3);
    return;
  }
  let lo = Math.min(...arr), hi = Math.max(...arr);
  if (hi === lo) { hi += 1; lo -= 1; }
  const X = i => (i / (arr.length - 1)) * (w - 2) + 1;
  const Y = v => h - 2 - ((v - lo) / (hi - lo)) * (h - 4);
  /* zero line if straddling */
  if (lo < 0 && hi > 0) {
    ctx.strokeStyle = "rgba(110,130,170,0.35)";
    ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
  }
  ctx.beginPath();
  arr.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
  ctx.strokeStyle = color || "#5b8cff"; ctx.lineWidth = 1.3; ctx.stroke();
  ctx.beginPath(); ctx.arc(X(arr.length - 1), Y(arr[arr.length - 1]), 2, 0, Math.PI * 2);
  ctx.fillStyle = color || "#5b8cff"; ctx.fill();
}

function accPct(arr) {
  if (!arr || !arr.length) return null;
  const pos = arr.filter(x => x > 0).length;
  const neg = arr.filter(x => x < 0).length;
  const n = pos + neg;
  if (!n) return null;
  return Math.round((pos / n) * 100);
}

function fpPanelHTML(id, name, value, weightTxt, acc, contrib, cls) {
  const v = value == null ? "—" : (value >= 0 ? "+" : "") + (+value).toFixed(3);
  const w = weightTxt == null ? "ω —" : weightTxt;
  const a = acc == null ? "acc —" : `acc ${acc >= 0 ? "+" : ""}${acc}%`;
  const c = contrib == null ? "Δ —" : `Δ ${contrib >= 0 ? "+" : ""}${(+contrib).toFixed(4)}`;
  return `<div class="fpanel">
    <div class="fp-top"><span class="fp-id">${id}</span><span class="fp-name">${name}</span></div>
    <div class="fp-val ${cls || (value >= 0 ? "pos" : "neg")}">${v}</div>
    <canvas class="fp-spark"></canvas>
    <div class="fp-meta"><span class="fp-w">${w}</span><span class="fp-acc ${acc != null && acc >= 50 ? "good" : acc != null && acc < 40 ? "bad" : ""}">${a}</span><span class="fp-ct">${c}</span></div>
  </div>`;
}

function renderPanels(snap) {
  const fh = stateExtras.fh;
  if (!fh) return;
  const { sym, tf } = state.detail;
  const key = `${sym}|${tf}`;
  const fv = snap && snap.formula_values && snap.formula_values[key];
  const sys = snap && snap.systems && snap.systems[key];

  /* --- core 31 --- */
  const coreEl = document.getElementById("panels-core");
  if (coreEl) {
    const names = fh.names.core || {};
    const weights = fh.weights || {};
    const wSum = Object.entries(weights)
      .filter(([k]) => +k <= 31)
      .reduce((a, [, b]) => a + (+b), 0) || 1;
    let html = "";
    const sparks = [];
    for (let f = 1; f <= 31; f++) {
      let value = null;
      if (fv && fv.adj && fv.adj[String(f)] != null) value = fv.adj[String(f)];
      else if (fv && fv.extra && fv.extra[String(f)] != null) value = fv.extra[String(f)];
      else if (f === 21 && sys && sys.base_confidence != null) value = sys.base_confidence;
      else if (f === 29 && cyc0(snap, sym, tf) != null) value = cyc0(snap, sym, tf);
      const w = weights[String(f)];
      const wTxt = w == null ? null
        : `ω ${(+w).toFixed(3)} · ${((w / wSum) * 100).toFixed(1)}%`;
      const acc = accPct(fh.core_accuracy[String(f)]);
      const contrib = (w != null && value != null) ? w * value : null;
      html += fpPanelHTML(`F${f}`, names[String(f)] || "", value, wTxt, acc, contrib,
        value == null ? "" : value >= 0 ? "pos" : "neg");
      sparks.push(fh.core[String(f)]);
    }
    coreEl.innerHTML = html;
    coreEl.querySelectorAll(".fp-spark").forEach((cv, i) =>
      drawMini(cv, sparks[i], sparks[i] && sparks[i].length && sparks[i][sparks[i].length - 1] >= 0 ? "#2fd48f" : "#ff5d73"));
  }

  /* --- emotions 22 --- */
  const emoEl = document.getElementById("panels-emotion");
  if (emoEl) {
    const em = sys && sys.emotion;
    const wts = (em && em.weights) || [];
    let html = "";
    const sparks = [];
    for (let e = 1; e <= 22; e++) {
      const value = e === 22 ? (em && (em.cerc ?? em.aggregate)) : (em && em.emotions ? em.emotions[e] : null);
      const w = wts[e - 1];
      const wTxt = e <= 21 && w != null ? `ω ${(+w).toFixed(3)} · ${((w / 1.0) * 100).toFixed(1)}%` : "CERC Σ";
      const acc = accPct(fh.emotion_accuracy[String(e)]);
      const contrib = (e <= 21 && w != null && value != null) ? w * value : (e === 22 && value != null ? value : null);
      const meta = EMOTION_META[e];
      html += fpPanelHTML(`E${e} ${meta[0]}`, meta[1], value, wTxt, acc, contrib, value >= 0 ? "pos" : "neg");
      sparks.push(fh.emotion[String(e)]);
    }
    emoEl.innerHTML = html;
    emoEl.querySelectorAll(".fp-spark").forEach((cv, i) =>
      drawMini(cv, sparks[i], "#ffb02e"));
  }

  /* --- physics 22 (P1..P21 + P22 UFP) --- */
  const phEl = document.getElementById("panels-physics");
  if (phEl) {
    const ph = sys && sys.physics;
    const alpha = fh.alpha || {};
    const cpl = (ph && ph.couplings) || {};
    const tme = (ph && ph.tme) != null ? ph.tme : 1;
    let html = "";
    const sparks = [];
    for (let p = 1; p <= 22; p++) {
      let value = null, contrib = null;
      if (ph) {
        if (p === 22) { value = ph.ufp; contrib = ph.ufp; }
        else if (ph.values) value = ph.values[p];
      }
      if (p <= 21 && value != null && alpha[p] != null) {
        const adj = NON_DIR_P.has(p) ? 2 * value - 1 : value * tme;
        contrib = (+alpha[p]) * adj * ((cpl[p] != null ? cpl[p] : 1));
      }
      const wTxt = p <= 21 && alpha[p] != null
        ? `α ${(alpha[p] * 100).toFixed(1)}%` : "Σα → comp 99";
      const acc = accPct(fh.physics_accuracy[String(p)]);
      const name = (fh.names.physics && fh.names.physics[String(p)]) || "";
      html += fpPanelHTML(`P${p} ${name}`, specLabel(p), value, wTxt, acc, contrib, value >= 0 ? "pos" : "neg");
      sparks.push(fh.physics[String(p)]);
    }
    phEl.innerHTML = html;
    phEl.querySelectorAll(".fp-spark").forEach((cv, i) =>
      drawMini(cv, sparks[i], "#5b8cff"));
  }
}

function cyc0(snap, sym, tf) {
  /* F29 DLP placement: direction-based display when no numeric series */
  const c = snap && snap.assets[sym] && snap.assets[sym].tfs[tf];
  if (!c || !c.issued) return 0;
  return c.direction > 0 ? 1 : -1;
}

function specLabel(p) {
  const L = {
    1: "PSSA superposition", 2: "WFCP ground state", 3: "HPVU", 4: "QTPB barriers",
    5: "MEE", 6: "DRPT drag", 7: "BEOFC", 8: "FDPD", 9: "TME Carnot",
    10: "PIPP random walk", 11: "RGVF", 12: "SSBD order", 13: "GLPLO gravity",
    14: "CELG", 15: "DSMS", 16: "RTDMP", 17: "BBIR", 18: "TPTI topology",
    19: "GSPI gauge", 20: "PDOBV", 21: "QDPC decay", 22: "Unified Physics Field",
  };
  return L[p] || "";
}

/* ---------------- System Co-relation network ---------------- */
function renderCorel(snapIn) {
  const cv = document.getElementById("corel-canvas");
  if (!cv) return;
  const snap = snapIn || state.snap;
  const { sym, tf } = state.detail;
  const sys = snap && snap.systems && snap.systems[`${sym}|${tf}`];
  const corel = sys && sys.corelation;
  const stats = document.getElementById("corel-stats");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (!corel || !corel.nodes) {
    ctx.fillStyle = "#8ea0c4"; ctx.font = "12px system-ui";
    ctx.fillText("awaiting first GRC — correlation needs ≥ 3 passes…", 16, 28);
    return;
  }

  /* fixed radial layout by kind */
  const POS = {
    "CORE": [0.50, 0.46],
    "EMOTION": [0.74, 0.26], "PHYSICS": [0.76, 0.68],
    "DB-1": [0.24, 0.16], "DB-2": [0.16, 0.37], "DB-3": [0.16, 0.63], "DB-4": [0.24, 0.84],
    "AGENT-G": [0.46, 0.90], "AGENT-L": [0.60, 0.90],
  };
  const KIND_COLOR = {
    brain: "#ffb02e", emotion: "#e879f9", physics: "#5b8cff",
    core: "#2fd48f", agent: "#5f719a",
  };
  const nodes = {};
  corel.nodes.forEach(n => {
    const p = POS[n.id] || [0.5, 0.5];
    nodes[n.id] = { ...n, x: p[0] * w, y: p[1] * (h - 20) + 10 };
  });

  /* edges */
  let strong = 0;
  (corel.edges || []).forEach(e => {
    const A = nodes[e.a], B = nodes[e.b];
    if (!A || !B) return;
    const r = e.corr;
    if (Math.abs(r) >= 0.6) strong++;
    const mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2 - Math.abs(A.x - B.x) * 0.08;
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.quadraticCurveTo(mx, my, B.x, B.y);
    ctx.strokeStyle = r >= 0 ? `rgba(47,212,143,${0.25 + Math.abs(r) * 0.6})`
                             : `rgba(255,93,115,${0.25 + Math.abs(r) * 0.6})`;
    ctx.lineWidth = 1 + Math.abs(r) * 6;
    ctx.stroke();
    /* mid label for strong links */
    if (Math.abs(r) >= 0.75) {
      ctx.fillStyle = "#c9d4ee"; ctx.font = "9px system-ui"; ctx.textAlign = "center";
      ctx.fillText(r.toFixed(2), mx, my - 4);
    }
  });

  /* nodes */
  for (const id in nodes) {
    const n = nodes[id];
    const R = n.id === "CORE" ? 30 : n.kind === "brain" ? 24 : n.kind === "agent" ? 16 : 26;
    ctx.beginPath(); ctx.arc(n.x, n.y, R, 0, Math.PI * 2);
    ctx.fillStyle = (KIND_COLOR[n.kind] || "#5f719a") + (n.ready === false ? "44" : "cc");
    ctx.fill();
    ctx.strokeStyle = "#0a1020"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#e8eefc"; ctx.font = "600 10px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(n.id, n.x, n.y + 3);
    if (n.ready === false) {
      ctx.fillStyle = "#5f719a"; ctx.font = "8px system-ui";
      ctx.fillText("Part 4", n.x, n.y + R + 10);
    }
  }

  if (stats) {
    stats.innerHTML =
      `<span>pair: <b>${sym} ${tf.toUpperCase()}</b></span>` +
      `<span>nodes: <b>${corel.nodes.length}</b></span>` +
      `<span>edges (window ≥3 passes): <b>${(corel.edges || []).length}</b></span>` +
      `<span>strong |r| ≥ 0.6: <b>${strong}</b></span>` +
      `<span>window: <b>50 GRCs</b></span>` +
      `<span class="lg-edge ex">r &gt; 0</span><span class="lg-edge in">r &lt; 0</span>`;
  }
}

function wireCorelHover() {
  const cv = document.getElementById("corel-canvas");
  const tip = document.getElementById("corel-tip");
  if (!cv || !tip) return;
  cv.addEventListener("mousemove", e => {
    const snap = state.snap;
    const { sym, tf } = state.detail;
    const sys = snap && snap.systems && snap.systems[`${sym}|${tf}`];
    const corel = sys && sys.corelation;
    if (!corel) { tip.classList.add("hidden"); return; }
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const POS = {
      "CORE": [0.50, 0.46], "EMOTION": [0.74, 0.26], "PHYSICS": [0.76, 0.68],
      "DB-1": [0.24, 0.16], "DB-2": [0.16, 0.37], "DB-3": [0.16, 0.63], "DB-4": [0.24, 0.84],
      "AGENT-G": [0.46, 0.90], "AGENT-L": [0.60, 0.90],
    };
    let hit = null;
    for (const n of corel.nodes) {
      const p = POS[n.id] || [0.5, 0.5];
      const x = p[0] * rect.width, y = p[1] * (rect.height - 20) + 10;
      if ((mx - x) ** 2 + (my - y) ** 2 < 34 ** 2) { hit = n; break; }
    }
    if (!hit) { tip.classList.add("hidden"); return; }
    const links = (corel.edges || []).filter(e => e.a === hit.id || e.b === hit.id);
    tip.innerHTML = `<b>${hit.id}</b>` + (links.length
      ? links.map(e => `<div>${e.a === hit.id ? e.b : e.a}: <span class="${e.corr >= 0 ? "pos" : "neg"}">${e.corr >= 0 ? "+" : ""}${e.corr.toFixed(3)}</span></div>`).join("")
      : `<div>no edges yet (needs ≥3 GRC passes)</div>`);
    tip.style.left = Math.min(rect.width - 170, mx + 14) + "px";
    tip.style.top = (my + 14) + "px";
    tip.classList.remove("hidden");
  });
  cv.addEventListener("mouseleave", () => tip.classList.add("hidden"));
}

/* ---------------- News & HF tab ---------------- */
async function ensureNews(force) {
  const sym = state.detail.sym;
  const fresh = Date.now() - stateExtras.newsAt < 45000;
  if (!force && stateExtras.newsFor === sym && fresh && stateExtras.news) return;
  try {
    const r = await fetch(`/api/news/${sym}`);
    if (!r.ok) return;
    const j = await r.json();
    stateExtras.news = j;
    stateExtras.newsFor = sym;
    stateExtras.newsAt = Date.now();
    renderNewsTab(state.snap);
  } catch (e) { /* retry */ }
}

function renderNewsTab(snap) {
  const el = document.getElementById("news-tab-content");
  if (!el) return;
  const nd = stateExtras.news;
  const { sym, tf } = state.detail;
  const key = `${sym}|${tf}`;
  const sys = snap && snap.systems && snap.systems[key];
  const meta = (snap && snap.news_meta && snap.news_meta.categories) || {};
  if (!nd && !sys) { el.innerHTML = `<div class="fm-item">awaiting data…</div>`; return; }
  const agg = (nd && nd.value) != null ? nd : (sys && sys.news);
  if (!agg) { el.innerHTML = `<div class="fm-item">awaiting data…</div>`; return; }

  /* category breakdown from live items */
  const byCat = {};
  (agg.items || []).forEach(it => {
    byCat[it.category] = byCat[it.category] || { n: 0, sum: 0, dur: it.duration };
    byCat[it.category].n++;
    byCat[it.category].sum += it.impact;
  });
  const catRows = Object.keys(meta).map(c => {
    const m = meta[c], b = byCat[c] || { n: 0, sum: 0, dur: m.duration };
    return `<tr class="${b.n ? "on" : ""}"><td>${c}</td>
      <td>${m.weight.toFixed(2)}</td><td>${m.duration} min</td>
      <td>${b.n}</td>
      <td class="${b.sum >= 0 ? "pos" : "neg"}">${b.sum ? (b.sum >= 0 ? "+" : "") + b.sum.toFixed(4) : "—"}</td></tr>`;
  }).join("");

  const tfRows = [["tf1", "22 min"], ["tf2", "44 min"], ["tf3", "130 min"], ["tf4", "247 min"]]
    .map(([t, lbl]) => {
      const v = (agg.impacts && agg.impacts[t]) != null ? agg.impacts[t] : 0;
      const pct = Math.round(Math.abs(v) * 100);
      return `<div class="tf-impact">
        <span class="ti-lbl">${lbl} · ${t.toUpperCase()}</span>
        <span class="ti-bar"><span class="${v >= 0 ? "up" : "down"}" style="width:${pct}%"></span></span>
        <span class="ti-val ${v >= 0 ? "pos" : "neg"}">${v >= 0 ? "+" : ""}${(+v).toFixed(4)}</span>
      </div>`;
    }).join("");

  const itemRows = (agg.items || []).map(it => `
    <tr>
      <td class="ni-cat">${it.category}</td>
      <td class="ni-head">${it.direction != null ? (it.direction > 0 ? "▲" : "▼") : (it.impact >= 0 ? "▲" : "▼")} ${it.headline}</td>
      <td class="${it.impact >= 0 ? "pos" : "neg"}">${it.impact >= 0 ? "+" : ""}${(+it.impact).toFixed(4)}</td>
      <td>${(+it.age_min).toFixed(1)} / ${it.duration}</td>
      <td>${it.source}</td>
      <td>${it.reliability.toFixed(1)}</td>
    </tr>`).join("");

  el.innerHTML = `
    <div class="news-top">
      <div class="agg-box ${agg.value >= 0 ? "pos" : "neg"}">
        <label>SIGNED AGGREGATE (247 min, clamped ±1)</label>
        <b>${agg.value >= 0 ? "+" : ""}${(+agg.value).toFixed(4)}</b>
        <span>bull ${(agg.bull ?? 0).toFixed(3)} · bear ${(agg.bear ?? 0).toFixed(3)} · items ${agg.count ?? (agg.items || []).length}</span>
        <span class="src">source: ${agg.source || "sim"}</span>
      </div>
      <div class="tf-impacts">${tfRows}</div>
    </div>
    <div class="sub-h">Category table <span class="hint">weight × sentiment × exp(−D/halflife) × reliability</span></div>
    <table class="news-table"><thead><tr><th>category</th><th>weight</th><th>halflife D</th><th>active</th><th>scored</th></tr></thead>
      <tbody>${catRows}</tbody></table>
    <div class="sub-h">Live items <span class="hint">age in minutes / duration</span></div>
    <div class="news-scroll"><table class="news-table"><thead><tr><th>cat</th><th>headline</th><th>impact</th><th>age/D</th><th>source</th><th>rel</th></tr></thead>
      <tbody>${itemRows || `<tr><td colspan="6">no active headlines</td></tr>`}</tbody></table></div>`;
}

function renderHF(snap) {
  const el = document.getElementById("hf-content");
  if (!el || !snap) return;
  const { sym, tf } = state.detail;
  const sys = snap.systems && snap.systems[`${sym}|${tf}`];
  const hf = sys && sys.hedge;
  if (!hf) { el.innerHTML = `<div class="fm-item">awaiting data…</div>`; return; }
  const cells = Object.entries(hf).map(([k, v]) => {
    if (typeof v !== "number") return "";
    return `<div class="hf-cell ${v >= 0 ? "pos" : "neg"}">
      <label>${k.toUpperCase()}</label><b>${v >= 0 ? "+" : ""}${(+v).toFixed(4)}</b></div>`;
  }).join("");
  const w = snap.weights && snap.weights["101"];
  el.innerHTML = `<div class="hf-row">${cells}</div>
    <div class="hint">retail_flow = −H_signal · IFR = institutional absorption · DPAD = dark-pool print ·
    SMDI = tanh((IFR − retail)/2) · component 101 weight ${w != null ? (+w).toFixed(4) : "0.0800"} (init 0.08, η-learned)</div>`;
}

/* ---------------- brain 3D glue ---------------- */
let actTimer = null;
function refreshActivations(force) {
  const now = Date.now();
  if (!force && now - stateExtras.actAt < 8000) return;
  stateExtras.actAt = now;
  const { sym, tf } = state.detail;
  clearTimeout(actTimer);
  actTimer = setTimeout(async () => {
    try {
      const r = await fetch(`/api/brain-activations?symbol=${sym}&tf=${tf}`);
      if (!r.ok) return;
      const j = await r.json();
      if (window.brain3D && window.brain3D.ready) window.brain3D.setActivations(j.per_instance);
    } catch (e) { /* retry next GRC */ }
  }, 60);
}

function renderBrain3dSide(snap) {
  const bars = document.getElementById("horm-bars");
  const stats = document.getElementById("brain3d-stats");
  if (!bars || !snap) return;
  const { sym, tf } = state.detail;
  const sys = snap.systems && snap.systems[`${sym}|${tf}`];
  const br = sys && sys.brain;
  const info = snap.brains || {};
  const conn = info.connectome || {};
  if (!br) {
    bars.innerHTML = `<div class="fm-item">awaiting first GRC…</div>`;
    stats.innerHTML = "";
    return;
  }
  /* hormone average bars */
  const hs = Object.entries(br.hormones_avg || {});
  bars.innerHTML = `<div class="sub-h">Hormone flow (4-instance average) · ${sym} ${tf.toUpperCase()}</div>` +
    hs.map(([k, v]) => {
      const p = Math.round(Math.max(0, Math.min(1, (+v))) * 100);
      return `<div class="horm-bar"><span class="hb-lbl">I_${k}</span>
        <span class="hb-track"><span style="width:${p}%"></span></span>
        <span class="hb-val">${(+v).toFixed(3)}</span></div>`;
    }).join("") +
    `<div class="consensus-line">consensus ${br.consensus >= 0 ? "+" : ""}${(+br.consensus).toFixed(4)}
      · strength ${(br.strength || 0).toFixed(3)} · agreement ${((br.agreement || 0) * 100).toFixed(0)}%</div>`;

  stats.innerHTML = `<div class="sub-h">Instances</div>` +
    `<table class="brain-table"><thead><tr><th>unit</th><th>MBON H</th><th>H_signal</th><th>EQ</th><th>dominant</th></tr></thead><tbody>` +
    (br.instances || []).map(i => `<tr><td><b>${i.id}</b></td>
      <td>${(i.hormone_state || 0).toFixed(3)}</td>
      <td class="${(i.h_signal || 0) >= 0 ? "pos" : "neg"}">${(i.h_signal || 0) >= 0 ? "+" : ""}${(i.h_signal || 0).toFixed(4)}</td>
      <td>${(i.experience || 0).toFixed(3)}</td>
      <td>${i.dominant || "—"}</td></tr>`).join("") +
    `</tbody></table>
    <div class="conn-line"><span>geometry: <b>${conn.units_per_brain ?? "?"} units</b></span>
      <span>${conn.edges != null ? conn.edges.toLocaleString() : "?"} edges</span>
      <span>source ${conn.source || "synthetic (spec populations)"}</span></div>`;
}

/* ---------------- wiring / intervals ---------------- */
function bootPart3() {
  wireTabs();
  wireExtraSelectors();
  wireCorelHover();
  renderGauges(state.snap);
  renderPanels(state.snap);
  const cr = document.getElementById("chart-refresh");
  if (cr) cr.addEventListener("click", () => ensureBars(true));
  /* chart bars: fetch at boot, refresh every 15 s and on every GRC */
  ensureBars(true);
  setInterval(() => ensureBars(), 15000);
  /* lazy refresh of heavy endpoints while their tab is open */
  setInterval(() => {
    if (stateExtras.tab === "formulas") ensureFH();
    if (stateExtras.tab === "news") ensureNews();
  }, 20000);
  window.addEventListener("resize", () => {
    drawChart(); renderGauges(state.snap);
    if (stateExtras.tab === "corel") renderCorel();
  });
  /* first activations once the brain module is ready */
  const waitBrain = tries => {
    if (window.brain3D && window.brain3D.ready) { refreshActivations(true); return; }
    if (tries > 0) setTimeout(() => waitBrain(tries - 1), 500);
  };
  waitBrain(20);
}
bootPart3();
