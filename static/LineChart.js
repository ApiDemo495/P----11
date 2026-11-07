/* ═══════════════════════════════════════════════════════════════════
   PriceChart — one line-chart engine for the whole app (OWN main chart,
   sub-timeframe cards, sparklines, and the probability/fan chart).

   Built from the research in docs/line-chart-research.md. The rules it
   implements, in order:
   1. The price line is the hero: drawn last-but-one, thickest stroke,
      last-point dot + dashed last-price line + filled axis tag.
   2. Axes use Heckbert "nice numbers" (1/2/5×10^k) with precision
      derived from the tick step; time ticks come from a step ladder.
      Zero baseline is NOT forced for prices (line-chart convention).
   3. Direct labelling, no legends: levels are tagged on their lines,
      overlays are labelled at their right end.
   4. The forecast fan never sets the y-range (it is clipped to the
      price-led range); history solid, forecast median dashed, a "NOW"
      divider between fact and projection (Bank of England fan rules).
   5. Volume gets its own sub-pane (≈16% height, 50% alpha) so it can
      never compress the price pane.
   6. Crosshair snaps (magnet) to the nearest real point and reads out
      time · price · Δ vs first visible point · volume.
   7. Missing data breaks the path — gaps are shown as gaps.
   8. Points are LTTB-downsampled to ≈ plot width; rendering is HiDPI
      (devicePixelRatio, capped at 2); the canvas carries an ARIA
      summary for screen readers.
   ═══════════════════════════════════════════════════════════════════ */
"use strict";

const PC_UP = "#26a69a", PC_DN = "#ef5350", PC_GOLD = "#e6b93d";

/* ── Heckbert nice numbers: step of 1/2/5×10^k covering span/target ── */
function pcTickStep(span, target) {
  if (!(span > 0) || !isFinite(span)) return 1;
  const raw = span / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const mult = norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1;
  return mult * mag;
}
function pcDecimals(step) {
  return Math.min(8, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
}
function pcFmt(v, dec) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  const s = a >= 1000 ? v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec })
                      : v.toFixed(dec);
  return s;
}
function pcFmtVol(v) {
  if (!isFinite(v)) return "";
  if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return String(Math.round(v));
}

/* ── LTTB downsampling (Steinarsson): preserve shape at pixel budget ── */
function pcLTTB(pts, threshold) {
  const n = pts.length;
  if (threshold >= n || threshold <= 2) return pts;
  const out = [pts[0]];
  const every = (n - 2) / (threshold - 2);
  let a = 0;
  for (let i = 0; i < threshold - 2; i++) {
    const r0 = Math.floor((i + 1) * every) + 1;
    const r1 = Math.min(Math.floor((i + 2) * every) + 1, n);
    let ax = 0, ay = 0, cnt = 0;
    for (let j = r0; j < r1; j++) { ax += pts[j].t; ay += pts[j].p; cnt++; }
    if (cnt) { ax /= cnt; ay /= cnt; }
    const o0 = Math.floor(i * every) + 1, o1 = Math.floor((i + 1) * every) + 1;
    const A = pts[a];
    let best = o0, bestArea = -1;
    for (let j = o0; j < o1; j++) {
      const area = Math.abs((A.t - ax) * (pts[j].p - A.p) - (A.t - pts[j].t) * (ay - A.p));
      if (area > bestArea) { bestArea = area; best = j; }
    }
    out.push(pts[best]); a = best;
  }
  out.push(pts[n - 1]);
  return out;
}

/* ── time axis: step ladder + span-aware format ── */
const PC_TSTEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 10800, 21600, 43200, 86400, 604800];
function pcTimeTicks(t0, t1, maxLabels) {
  const span = Math.max(1, t1 - t0);
  let step = PC_TSTEPS[PC_TSTEPS.length - 1];
  for (const s of PC_TSTEPS) { if (span / s <= maxLabels) { step = s; break; } }
  const out = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) out.push(t);
  return { ticks: out, step };
}
function pcFmtTime(t, span) {
  const d = new Date(t * 1000);
  const hh = String(d.getHours()).padStart(2, "0"), mm = String(d.getMinutes()).padStart(2, "0"),
        ss = String(d.getSeconds()).padStart(2, "0");
  if (span > 129600) return `${d.getDate()} ${d.toLocaleString("en-US", { month: "short" })} ${hh}:${mm}`;
  if (span < 240) return `${hh}:${mm}:${ss}`;
  return `${hh}:${mm}`;
}

class PriceChart {
  constructor(el, opts = {}) {
    this.el = el;
    this.spark = !!opts.spark;
    this.histogram = !!opts.histogram;
    this.cv = document.createElement("canvas");
    this.cv.style.cssText = "display:block;width:100%;height:100%;touch-action:pan-y";
    this.cv.setAttribute("role", "img");
    el.appendChild(this.cv);
    this.model = null;
    this._lines = [];
    this.cross = null;
    const upd = e => {
      const r = this.cv.getBoundingClientRect();
      this.cross = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.draw();
    };
    this.cv.addEventListener("pointermove", upd);
    this.cv.addEventListener("pointerdown", upd);
    this.cv.addEventListener("pointerleave", () => { this.cross = null; this.draw(); });
  }

  /* ── data entry points (kept compatible with the app's call sites) ── */
  setData(bars) { // OHLC-ish bars → close series (+ volume pane)
    const pts = (bars || [])
      .map(b => ({ t: b.time > 1e12 ? b.time / 1000 : b.time, p: b.close, v: b.v || 0 }))
      .filter(p => Number.isFinite(p.t) && p.p > 0);
    this._basePts = pts;
    this._emit();
  }
  tick(p) {
    if (!this._basePts || !this._basePts.length || !(p > 0)) return;
    const now = Date.now() / 1000, last = this._basePts[this._basePts.length - 1];
    if (now - last.t < 1) last.p = p; else this._basePts.push({ t: now, p });
    if (this._basePts.length > 1200) this._basePts.splice(0, this._basePts.length - 1200);
    this._emit();
  }
  setLines(lines) { // [{price, color, title, dash}] — quad-card levels
    this._lines = (lines || []).filter(L => Number.isFinite(L.price))
      .map(L => ({ p: L.price, label: L.title || "", color: L.color || PC_GOLD, dash: !!L.dash }));
    this._emit();
  }
  _emit() {
    if (!this._basePts) return;
    this.setModel({
      series: [{ pts: this._basePts }],
      levels: this._lines,
      ma: this.spark ? 0 : 20,
      t0: this._basePts[0]?.t, t1: this._basePts[this._basePts.length - 1]?.t,
      ariaLabel: this.el?.id ? `${this.el.id} price chart` : "price chart",
    });
  }
  setModel(model) { this.model = model; this.draw(); }

  /* ── the draw ── */
  draw() {
    const m = this.model; if (!m) return;
    const W = this.el.clientWidth || 300, H = this.el.clientHeight || 120;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (this.cv.width !== Math.round(W * dpr) || this.cv.height !== Math.round(H * dpr)) {
      this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr);
    }
    const g = this.cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const ink = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#e8e8f2";
    const dim = "#9a9ab2";

    const series = (m.series || [])
      .map(s => ({ ...s, pts: (s.pts || []).filter(p => Number.isFinite(p.t) && Number.isFinite(p.p) && p.p > 0)
                            .sort((a, b) => a.t - b.t) }))
      .filter(s => s.pts.length);
    const main = series[0];
    const spark = this.spark;

    /* ── empty state ── */
    if (!main || main.pts.length < 2) {
      if (!spark) {
        g.fillStyle = dim; g.font = "11px ui-monospace,monospace"; g.textAlign = "center";
        g.fillText(m.emptyText || "waiting for data…", W / 2, H / 2);
        g.textAlign = "left";
      }
      return;
    }

    /* ── layout ── */
    const padL = 4, padR = spark ? 2 : 62, padT = 6, padB = spark ? 2 : 18;
    const plotW = Math.max(20, W - padL - padR), plotH = Math.max(20, H - padT - padB);
    const hasVol = !spark && main.pts.some(p => (p.v || 0) > 0);
    const priceH = hasVol ? plotH * 0.84 : plotH;      // volume sub-pane below
    const volTop = padT + priceH, volH = plotH - priceH;

    /* ── domains ──
       Y: series + labelled levels only. The fan is NEVER in the domain
       (research rule 7: fans get clipped, prices set the range). */
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const p of s.pts) { lo = Math.min(lo, p.p); hi = Math.max(hi, p.p); }
    const levels = (m.levels || []).filter(L => Number.isFinite(L.p));
    /* DOMAIN RULE (TradingView "scale price chart only"; research
       addendum): the axis comes from the DATA plotted — price series
       plus the fan's inner band/median. Levels and the fan's outer
       band are annotations drawn ONTO that domain, never folded into
       it, or a TP 1.3% away compresses all price action flat.
       Off-scale levels render as edge tags with their real distance. */
    if (m.fan?.pts?.length) for (const f of m.fan.pts) {
      for (const k of ["lo2", "hi2", "mid"]) if (Number.isFinite(f[k])) { lo = Math.min(lo, f[k]); hi = Math.max(hi, f[k]); }
    }
    const compact = !spark && (plotW < 300 || priceH < 130);
    if (!(hi > lo)) { hi = lo * 1.001 + 1e-9; lo = lo * 0.999; }
    const yStep = pcTickStep(hi - lo, Math.max(2, Math.round(priceH / 52)));
    if (!spark) { lo = Math.floor(lo / yStep) * yStep; hi = Math.ceil(hi / yStep) * yStep; }
    const padY = (hi - lo) * 0.04; lo -= padY; hi += padY;
    const dec = pcDecimals(yStep);

    const allT = main.pts;
    let t0 = m.t0 ?? allT[0].t, t1 = m.t1 ?? allT[allT.length - 1].t;
    if (m.fan?.pts?.length) t1 = Math.max(t1, m.fan.pts[m.fan.pts.length - 1].t);
    if (!(t1 > t0)) t1 = t0 + 1;

    const X = t => padL + (t - t0) / (t1 - t0) * plotW;
    const Y = p => padT + (hi - p) / (hi - lo) * priceH;

    /* ── downsample to the pixel budget (LTTB) ── */
    const budget = Math.max(24, Math.floor(plotW));
    const drawPts = main.pts.length > budget ? pcLTTB(main.pts, budget) : main.pts;

    /* histogram spark (volume cards): bars from the baseline, no line */
    if (this.histogram) {
      let vmx = 0; for (const p of main.pts) vmx = Math.max(vmx, p.p);
      const bw = Math.max(1, plotW / drawPts.length - 0.5);
      drawPts.forEach((p, i) => {
        const hgt = Math.max(0.5, p.p / (vmx || 1) * (plotH - 2));
        g.fillStyle = (i && p.p < drawPts[i - 1].p) ? "rgba(239,83,80,.55)" : "rgba(38,166,154,.55)";
        g.fillRect(X(p.t) - bw / 2, padT + plotH - hgt, bw, hgt);
      });
      return;
    }

    /* ── grid + y axis (nice-number ticks) ── */
    if (!spark) {
      g.font = "10px ui-monospace,monospace";
      for (let v = Math.ceil(lo / yStep) * yStep; v <= hi; v += yStep) {
        const y = Y(v);
        g.strokeStyle = "rgba(140,140,170,.11)";
        g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
        g.fillStyle = dim; g.fillText(pcFmt(v, dec), padL + plotW + 5, y + 3);
      }
      /* ── time axis ── */
      const { ticks } = pcTimeTicks(t0, t1, Math.max(2, Math.floor(plotW / 86)));
      g.fillStyle = dim; g.textAlign = "center";
      for (const t of ticks) g.fillText(pcFmtTime(t, t1 - t0), X(t), H - 5);
      g.textAlign = "left";
    }

    /* ── clip everything data-shaped to the plot ── */
    g.save();
    g.beginPath(); g.rect(padL, padT - 2, plotW, plotH + 4); g.clip();

    /* future region + NOW divider (fact | projection) */
    if (m.nowT != null && m.nowT > t0 && m.nowT < t1) {
      const x = X(m.nowT);
      g.fillStyle = "rgba(130,130,210,.05)";
      g.fillRect(x, padT, padL + plotW - x, plotH);
      g.strokeStyle = "rgba(230,185,61,.5)"; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(x, padT); g.lineTo(x, padT + plotH); g.stroke(); g.setLineDash([]);
      if (!spark) { g.fillStyle = PC_GOLD; g.font = "bold 8px ui-monospace,monospace";
        g.fillText("NOW", Math.max(padL + 1, x - 22), padT + 8); g.font = "10px ui-monospace,monospace"; }
    }

    /* ── fan: nested band clipped to the price-led range; median dashed ── */
    if (m.fan?.pts?.length > 1) {
      const f = m.fan.pts, oT = m.fan.originT ?? f[0].t, oP = m.fan.originP ?? main.pts[main.pts.length - 1].p;
      g.beginPath(); g.moveTo(X(oT), Y(oP));
      f.forEach(p => g.lineTo(X(p.t), Y(p.hi)));
      for (let i = f.length - 1; i >= 0; i--) g.lineTo(X(f[i].t), Y(f[i].lo));
      g.closePath(); g.fillStyle = "rgba(38,166,154,.10)"; g.fill();
      if (f.some(p => Number.isFinite(p.lo2) && Number.isFinite(p.hi2))) {
        // inner band (middle ~50% of outcomes): darker = more likely
        g.beginPath(); g.moveTo(X(oT), Y(oP));
        f.forEach(p => { if (Number.isFinite(p.hi2)) g.lineTo(X(p.t), Y(p.hi2)); });
        for (let i = f.length - 1; i >= 0; i--) if (Number.isFinite(f[i].lo2)) g.lineTo(X(f[i].t), Y(f[i].lo2));
        g.closePath(); g.fillStyle = "rgba(38,166,154,.17)"; g.fill();
      }
      g.beginPath(); g.moveTo(X(oT), Y(oP));
      f.forEach(p => g.lineTo(X(p.t), Y(p.hi)));
      g.strokeStyle = "rgba(38,166,154,.45)"; g.lineWidth = 1; g.stroke();
      g.beginPath(); g.moveTo(X(oT), Y(oP));
      f.forEach(p => g.lineTo(X(p.t), Y(p.lo)));
      g.strokeStyle = "rgba(239,83,80,.45)"; g.stroke();
      if (f.some(p => Number.isFinite(p.mid))) {
        g.beginPath(); g.moveTo(X(oT), Y(oP));
        f.forEach(p => { if (Number.isFinite(p.mid)) g.lineTo(X(p.t), Y(p.mid)); });
        g.strokeStyle = "#7fd8cf"; g.setLineDash([4, 3]); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]);
      }
      g.lineWidth = 1;
    }

    /* ── volume sub-pane (own scale; never touches the price range) ── */
    if (hasVol) {
      let vmax = 0; for (const p of main.pts) vmax = Math.max(vmax, p.v || 0);
      const bw = Math.min(12, Math.max(1, plotW / drawPts.length - 0.5));
      drawPts.forEach((p, i) => {
        const hgt = Math.max(0.5, (p.v || 0) / (vmax || 1) * (volH - 6));
        g.fillStyle = (i && p.p < drawPts[i - 1].p) ? "rgba(239,83,80,.5)" : "rgba(38,166,154,.5)";
        g.fillRect(X(p.t) - bw / 2, volTop + volH - hgt, bw, hgt);
      });
      g.fillStyle = dim; g.font = "8px ui-monospace,monospace";
      g.fillText("VOL " + pcFmtVol(vmax), padL + 1, volTop + 8);
      g.font = "10px ui-monospace,monospace";
    }

    /* ── levels: in-range = line + direct tag; off-scale = edge tag ── */
    const offScale = [];
    for (const L of levels) {
      if (L.p < lo || L.p > hi) { offScale.push(L); continue; }
      const y = Y(L.p);
      g.strokeStyle = L.color; g.globalAlpha = .85;
      g.setLineDash(L.dash ? [6, 4] : []);
      g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;
      if (!spark && L.label) {
        g.font = "bold 9px ui-monospace,monospace";
        g.fillRect(padL + plotW + 1, y - 8, padR - 3, 15);
        g.fillStyle = "#0b0b14";
        g.fillText(L.label, padL + plotW + 4, y + 3);
        if (!compact) { g.fillStyle = L.color; g.fillText(`${L.label} ${pcFmt(L.p, dec)}`, padL + 4, y - 3); }
        g.font = "10px ui-monospace,monospace";
      }
    }

    /* ── series: subtle area, then the hero stroke; gaps break the path ── */
    const seriesColor = main.color || (main.pts[main.pts.length - 1].p >= main.pts[0].p ? PC_UP : PC_DN);
    const medianDt = (() => {
      if (main.pts.length < 3) return 0;
      const dts = [];
      for (let i = 1; i < main.pts.length; i++) dts.push(main.pts[i].t - main.pts[i - 1].t);
      return dts.sort((a, b) => a - b)[Math.floor(dts.length / 2)] || 0;
    })();
    const trace = (pts, closeArea) => {
      g.beginPath();
      let pen = false, prevT = null;
      pts.forEach(p => {
        const x = X(p.t), y = Y(p.p);
        if (!pen || (medianDt && prevT != null && p.t - prevT > 3 * medianDt)) {
          if (closeArea && pen) { /* area closes per-segment below */ }
          g.moveTo(x, y); pen = true;
        } else g.lineTo(x, y);
        prevT = p.t;
      });
    };
    if (!spark) { // area fill (segment-wise so gaps stay honest)
      const grad = g.createLinearGradient(0, padT, 0, padT + priceH);
      const base = seriesColor === PC_UP ? "38,166,154" : seriesColor === PC_DN ? "239,83,80" : "226,226,245";
      grad.addColorStop(0, `rgba(${base},.20)`); grad.addColorStop(1, `rgba(${base},0)`);
      g.beginPath();
      let pen = false, prevT = null, segStart = null;
      drawPts.forEach(p => {
        const x = X(p.t), y = Y(p.p);
        if (!pen || (medianDt && prevT != null && p.t - prevT > 3 * medianDt)) {
          if (pen && segStart) { g.lineTo(X(prevT), padT + priceH); g.lineTo(X(segStart), padT + priceH); g.closePath(); }
          g.moveTo(x, y); pen = true; segStart = p.t;
        } else g.lineTo(x, y);
        prevT = p.t;
      });
      if (pen && segStart) { g.lineTo(X(prevT), padT + priceH); g.lineTo(X(segStart), padT + priceH); g.closePath(); }
      g.fillStyle = grad; g.fill();
    }
    trace(drawPts, false);
    g.strokeStyle = seriesColor; g.lineWidth = spark ? 1.5 : 2;
    g.lineJoin = "round"; g.lineCap = "round"; g.stroke(); g.lineWidth = 1;

    /* ── moving-average overlay (muted, thinner, end-labelled) ── */
    const maN = m.ma | 0;
    if (!spark && maN >= 2 && main.pts.length >= maN) {
      let acc = 0; const maPts = [];
      main.pts.forEach((p, i) => {
        acc += p.p; if (i >= maN) acc -= main.pts[i - maN].p;
        if (i >= maN - 1) maPts.push({ t: p.t, p: acc / maN });
      });
      g.beginPath();
      maPts.forEach((p, i) => i ? g.lineTo(X(p.t), Y(p.p)) : g.moveTo(X(p.t), Y(p.p)));
      g.strokeStyle = "rgba(235,235,250,.55)"; g.lineWidth = 1; g.stroke();
      const lp2 = maPts[maPts.length - 1];
      if (!compact) {
        g.fillStyle = "rgba(235,235,250,.75)"; g.font = "8px ui-monospace,monospace";
        g.fillText("MA" + maN, Math.min(padL + plotW - 24, X(lp2.t) + 3), Y(lp2.p) - 3);
        g.font = "10px ui-monospace,monospace";
      }
    }

    /* ── overlays passed in by the caller (already computed series) ── */
    for (const ov of (m.overlays || [])) {
      if (!ov.pts?.length) continue;
      g.beginPath();
      ov.pts.forEach((p, i) => i ? g.lineTo(X(p.t), Y(p.p)) : g.moveTo(X(p.t), Y(p.p)));
      g.strokeStyle = ov.color || "rgba(235,235,250,.55)";
      g.setLineDash(ov.dash ? [4, 3] : []); g.stroke(); g.setLineDash([]);
      if (ov.label && !spark) {
        const e = ov.pts[ov.pts.length - 1];
        g.fillStyle = ov.color || "#e8e8f2"; g.font = "8px ui-monospace,monospace";
        g.fillText(ov.label, Math.min(padL + plotW - 24, X(e.t) + 3), Y(e.p) - 3);
        g.font = "10px ui-monospace,monospace";
      }
    }

    /* ── last point: dot + dashed last-price line + axis tag ── */
    const lp = main.pts[main.pts.length - 1], lx = X(lp.t), ly = Y(lp.p);
    g.fillStyle = seriesColor; g.beginPath(); g.arc(lx, ly, spark ? 2 : 3, 0, 7); g.fill();
    if (!spark) {
      g.strokeStyle = seriesColor; g.globalAlpha = .5; g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(padL, ly); g.lineTo(padL + plotW, ly); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;
    }
    g.restore(); // un-clip for axis tags

    if (!spark) {
      // off-scale levels: pinned to the plot edge they lie beyond,
      // with their real distance from the last price (never hidden,
      // never allowed to compress the data range)
      for (const L of offScale) {
        const above = L.p > hi, ey = above ? padT + 8 : padT + priceH - 8;
        g.fillStyle = L.color;
        g.beginPath();
        if (above) { g.moveTo(padL + plotW - 5, ey + 3); g.lineTo(padL + plotW + 5, ey + 3); g.lineTo(padL + plotW, ey - 3); }
        else { g.moveTo(padL + plotW - 5, ey - 3); g.lineTo(padL + plotW + 5, ey - 3); g.lineTo(padL + plotW, ey + 3); }
        g.closePath(); g.fill();
        g.fillRect(padL + plotW + 1, ey - 8, padR - 2, 15);
        g.fillStyle = "#0b0b14"; g.font = "bold 9px ui-monospace,monospace";
        g.fillText(L.label || "", padL + plotW + 4, ey + 3);
        if (!compact && L.label) {
          const dist = lp.p ? (L.p / lp.p - 1) * 100 : 0;
          g.fillStyle = L.color; g.font = "8px ui-monospace,monospace"; g.textAlign = "right";
          g.fillText(`${L.label} ${pcFmt(L.p, dec)} ${above ? "▲" : "▼"} ${dist >= 0 ? "+" : ""}${dist.toFixed(2)}%`,
                     padL + plotW - 12, above ? ey + 4 : ey + 3);
          g.textAlign = "left";
        }
        g.font = "10px ui-monospace,monospace";
      }
      g.fillStyle = seriesColor;
      g.fillRect(padL + plotW + 1, ly - 9, padR - 2, 18);
      g.fillStyle = "#0b0b14"; g.font = "bold 10px ui-monospace,monospace";
      g.fillText(pcFmt(lp.p, dec), padL + plotW + 4, ly + 3.5);
      /* H/L direct labels (selective annotation of extremes) */
      let hp = main.pts[0], lpLow = main.pts[0];
      for (const p of main.pts) { if (p.p > hp.p) hp = p; if (p.p < lpLow.p) lpLow = p; }
      g.fillStyle = dim; g.font = "8px ui-monospace,monospace";
      if (!compact && hp !== lp && hp.t !== lpLow.t) {
        g.fillText("H " + pcFmt(hp.p, dec), Math.min(padL + plotW - 52, Math.max(padL, X(hp.t) - 8)), Math.max(8, Y(hp.p) - 4));
        g.fillText("L " + pcFmt(lpLow.p, dec), Math.min(padL + plotW - 52, Math.max(padL, X(lpLow.t) - 8)), Math.min(padT + priceH - 2, Y(lpLow.p) + 10));
      }
      g.font = "10px ui-monospace,monospace";
    }

    /* ── crosshair: magnet-snapped to the nearest real point ── */
    if (this.cross && !spark) {
      const { x, y } = this.cross;
      if (x >= padL && x <= padL + plotW && y >= padT && y <= padT + plotH) {
        const tAt = t0 + (x - padL) / plotW * (t1 - t0);
        let near = main.pts[0];
        for (const p of main.pts) if (Math.abs(p.t - tAt) < Math.abs(near.t - tAt)) near = p;
        const nx = X(near.t), ny = Y(near.p);
        g.strokeStyle = "rgba(210,210,240,.4)"; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(nx, padT); g.lineTo(nx, padT + plotH);
        g.moveTo(padL, ny); g.lineTo(padL + plotW, ny); g.stroke(); g.setLineDash([]);
        g.fillStyle = "#fff"; g.beginPath(); g.arc(nx, ny, 3.2, 0, 7); g.fill();
        g.fillStyle = "#26263a"; g.fillRect(padL + plotW + 1, ny - 8, padR - 2, 16);
        g.fillStyle = "#fff"; g.font = "bold 9px ui-monospace,monospace";
        g.fillText(pcFmt(near.p, dec), padL + plotW + 4, ny + 3);
        const first = main.pts[0];
        const chg = first?.p ? (near.p / first.p - 1) * 100 : 0;
        const txt = `${pcFmtTime(near.t, t1 - t0)}${t1 - t0 < 240 ? "" : ""} · ${pcFmt(near.p, dec)} · ${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%` +
                    ((near.v || 0) > 0 ? ` · Vol ${pcFmtVol(near.v)}` : "");
        g.font = "9px ui-monospace,monospace";
        const tw = g.measureText(txt).width + 10;
        const bx = Math.min(Math.max(nx + 8, padL), padL + plotW - tw), by = Math.max(padT + 2, ny - 30);
        g.fillStyle = "rgba(16,16,28,.92)"; g.fillRect(bx, by, tw, 16);
        g.strokeStyle = "rgba(160,160,200,.35)"; g.strokeRect(bx, by, tw, 16);
        g.fillStyle = ink; g.fillText(txt, bx + 5, by + 11.5);
      }
    }

    /* ── ARIA summary (canvas is role=img; keep its label truthful) ── */
    const label = m.ariaLabel || "line chart";
    const first = main.pts[0];
    this.cv.setAttribute("aria-label",
      `${label}. Last ${pcFmt(lp.p, dec)} at ${pcFmtTime(lp.t, t1 - t0)}, ` +
      `range ${pcFmt(lo + padY, dec)} to ${pcFmt(hi - padY, dec)}, ` +
      `${main.pts.length} points from ${pcFmt(first.p, dec)}.`);
  }
}

/* global (loaded as a classic script before the app module) */
window.PriceChart = PriceChart;
