/* Drosophila Brain System — 3D visualization (Three.js).
 * 4 instances in a 2x2 grid, one shared WebGL context (scissor viewports).
 * - instanced spheres per unit (7,700 x 4), activation -> color gradient
 * - edge LineSegments colored by sign of the connectome weight
 * - white flash (500 ms fade) on units whose activation jumped at a GRC
 * - HTML region labels projected from region centers each frame
 * - drag to orbit, wheel to zoom
 */
import * as THREE from "./vendor/three.module.js";

const G = {
  units: 0, edges: 0,
  pos: null,       // Float32Array (units*3), normalized /1000
  pairs: null,     // Uint16Array (edges*2)
  weights: null,   // Int8Array (edges)
  regions: [],
  news: [],
};

const canvas = document.getElementById("brains-canvas");
const labelBox = document.getElementById("region-labels");
const fallback = document.getElementById("brains-fallback");

let renderer = null, visible = false, built = false, geoReady = false;
const scenes = [], cams = [], meshes = [], lineObjs = [];
const targetCol = [];            // Float32Array units*3 per instance
const flash = [new Map(), new Map(), new Map(), new Map()]; // idx -> deadline
const QUAD_LABELS = [];          // per instance: [{el, center}]
let yaw = 0.65, pitch = 0.34, dist = 2.45, raf = 0, labelTick = 0;
let drag = null;

function b64Bytes(s) {
  const bin = atob(s);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}

/* activation 0..255 -> color ramp */
const RAMP = [
  [0.00, 0x0a, 0x16, 0x30],
  [0.22, 0x16, 0x33, 0x7a],
  [0.45, 0x1e, 0x6f, 0xd9],
  [0.68, 0x2f, 0xd4, 0x8f],
  [0.86, 0xff, 0xd1, 0x66],
  [1.00, 0xff, 0xff, 0xff],
];
function rampColor(v, out) {
  const t = Math.max(0, Math.min(1, v / 255));
  let a = RAMP[0], b = RAMP[RAMP.length - 1];
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) { a = RAMP[i - 1]; b = RAMP[i]; break; }
  }
  const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
  out[0] = (a[1] + (b[1] - a[1]) * f) / 255;
  out[1] = (a[2] + (b[2] - a[2]) * f) / 255;
  out[2] = (a[3] + (b[3] - a[3]) * f) / 255;
}

async function loadGeometry() {
  const r = await fetch("/api/brain-geometry");
  if (!r.ok) throw new Error("geometry http " + r.status);
  const j = await r.json();
  G.units = j.units;
  G.edges = j.edges;
  const posRaw = new Int16Array(b64Bytes(j.positions).buffer);
  G.pos = new Float32Array(G.units * 3);
  for (let i = 0; i < G.units * 3; i++) G.pos[i] = posRaw[i] / 1000;
  G.pairs = new Uint16Array(b64Bytes(j.edge_pairs).buffer);
  G.weights = new Int8Array(b64Bytes(j.edge_weights).buffer);
  G.regions = j.regions || [];
  G.news = j.news_units || [];
  geoReady = true;
}

function buildScene(q) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070c1a);

  /* units: instanced low-poly spheres */
  const geo = new THREE.IcosahedronGeometry(1, 0);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const mesh = new THREE.InstancedMesh(geo, mat, G.units);
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
  const m4 = new THREE.Matrix4();
  const newsSet = new Set(G.news);
  for (let i = 0; i < G.units; i++) {
    const s = newsSet.has(i) ? 0.017 : 0.010;
    m4.makeScale(s, s, s);
    m4.setPosition(G.pos[i * 3], G.pos[i * 3 + 1], G.pos[i * 3 + 2]);
    mesh.setMatrixAt(i, m4);
    mesh.setColorAt(i, new THREE.Color(0x0a, 0x16, 0x30));
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  meshes.push(mesh);
  targetCol.push(new Float32Array(G.units * 3));

  /* edges: one LineSegments, vertex colors by sign/weight */
  const epos = new Float32Array(G.edges * 6);
  const ecol = new Float32Array(G.edges * 6);
  for (let e = 0; e < G.edges; e++) {
    const a = G.pairs[e * 2], b = G.pairs[e * 2 + 1];
    epos[e * 6] = G.pos[a * 3];     epos[e * 6 + 1] = G.pos[a * 3 + 1]; epos[e * 6 + 2] = G.pos[a * 3 + 2];
    epos[e * 6 + 3] = G.pos[b * 3]; epos[e * 6 + 4] = G.pos[b * 3 + 1]; epos[e * 6 + 5] = G.pos[b * 3 + 2];
    const w = G.weights[e];
    const mag = 0.30 + 0.70 * Math.min(1, Math.abs(w) / 127);
    const c = w >= 0 ? [0.16, 0.80, 0.50] : [0.92, 0.26, 0.36];
    for (let k = 0; k < 2; k++) {
      ecol[e * 6 + k * 3] = c[0] * mag;
      ecol[e * 6 + k * 3 + 1] = c[1] * mag;
      ecol[e * 6 + k * 3 + 2] = c[2] * mag;
    }
  }
  const egeo = new THREE.BufferGeometry();
  egeo.setAttribute("position", new THREE.BufferAttribute(epos, 3));
  egeo.setAttribute("color", new THREE.BufferAttribute(ecol, 3));
  const emat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.30, depthWrite: false,
  });
  const lines = new THREE.LineSegments(egeo, emat);
  scene.add(lines);
  lineObjs.push(lines);

  const cam = new THREE.PerspectiveCamera(40, 1, 0.05, 60);
  scenes.push(scene);
  cams.push(cam);
}

function makeLabels() {
  labelBox.innerHTML = "";
  QUAD_LABELS.length = 0;
  const captions = ["B1", "B2", "B3", "B4"];
  for (let q = 0; q < 4; q++) {
    const cap = document.createElement("div");
    cap.className = "quad-caption";
    cap.textContent = captions[q];
    labelBox.appendChild(cap);
    const items = G.regions.map(reg => {
      const el = document.createElement("div");
      el.className = "region-label";
      el.innerHTML = `<i style="background:${reg.color}"></i>${reg.name}`;
      el.title = `${reg.name} — ${reg.count} units`;
      labelBox.appendChild(el);
      return { el, center: new THREE.Vector3(reg.center[0], reg.center[1], reg.center[2]) };
    });
    QUAD_LABELS.push({ cap, items });
  }
}

function resize() {
  if (!renderer || !canvas.clientWidth) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(dpr);
}

function quadRect(q) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const hw = Math.floor(w / 2), hh = Math.floor(h / 2);
  /* three.js viewport origin: bottom-left. q0/q1 = top row. */
  const left = (q % 2) * hw;
  const bottom = (q < 2) ? hh : 0;
  return [left, bottom, hw, hh];
}

function updateFlashes(now) {
  for (let q = 0; q < 4; q++) {
    const map = flash[q];
    if (!map.size) continue;
    const mesh = meshes[q];
    const col = new THREE.Color();
    for (const [idx, deadline] of map) {
      const t = Math.max(0, Math.min(1, (deadline - now) / 500));
      if (t <= 0) {
        col.setRGB(targetCol[q][idx * 3], targetCol[q][idx * 3 + 1], targetCol[q][idx * 3 + 2]);
        map.delete(idx);
      } else {
        /* ease-out fade from white back to the activation color */
        const k = t * t;
        col.setRGB(
          targetCol[q][idx * 3] + (1 - targetCol[q][idx * 3]) * k,
          targetCol[q][idx * 3 + 1] + (1 - targetCol[q][idx * 3 + 1]) * k,
          targetCol[q][idx * 3 + 2] + (1 - targetCol[q][idx * 3 + 2]) * k,
        );
      }
      mesh.setColorAt(idx, col);
    }
    mesh.instanceColor.needsUpdate = true;
  }
}

function placeLabels() {
  if (!labelTick) {
    labelTick = 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const box = labelBox.getBoundingClientRect();
    const cvs = canvas.getBoundingClientRect();
    const v = new THREE.Vector3();
    for (let q = 0; q < 4; q++) {
      const [lx, lb, lw, lh] = quadRect(q);
      const { cap, items } = QUAD_LABELS[q];
      cap.style.left = (cvs.left - box.left + lx + 10) + "px";
      cap.style.top = (cvs.top - box.top + (q < 2 ? lh : lb) + (q < 2 ? 8 : lh - 26)) + "px";
      for (const it of items) {
        v.copy(it.center).project(cams[q]);
        const sx = lx + (v.x * 0.5 + 0.5) * lw;
        const sy = lb + (-v.y * 0.5 + 0.5) * lh;
        it.el.style.left = (cvs.left - box.left + sx) + "px";
        it.el.style.top = (cvs.top - box.top + sy) + "px";
        it.el.style.opacity = v.z < 1 ? "1" : "0.25";
      }
    }
  }
}

function frame() {
  raf = requestAnimationFrame(frame);
  if (!built || !visible || document.hidden) return;
  const now = performance.now();
  yaw += 0.0022;
  updateFlashes(now);
  resize();

  renderer.setScissorTest(true);
  for (let q = 0; q < 4; q++) {
    const [x, y, w, h] = quadRect(q);
    renderer.setViewport(x, y, w, h);
    renderer.setScissor(x, y, w, h);
    const cam = cams[q];
    cam.aspect = w / Math.max(1, h);
    cam.updateProjectionMatrix();
    cam.position.set(
      dist * Math.cos(pitch) * Math.sin(yaw),
      dist * Math.sin(pitch),
      dist * Math.cos(pitch) * Math.cos(yaw),
    );
    cam.lookAt(0, 0, 0);
    renderer.render(scenes[q], cam);
  }
  renderer.setScissorTest(false);
  labelTick = 0;
  placeLabels();
}

/* ---------------- public API (used by app.js) ---------------- */
export function initBrain3D() {
  if (!canvas) return;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  } catch (e) {
    if (fallback) fallback.classList.remove("hidden");
    return;
  }
  loadGeometry().then(() => {
    for (let q = 0; q < 4; q++) buildScene(q);
    makeLabels();
    built = true;
    resize();
    window.addEventListener("resize", () => { labelTick = 1; });
  }).catch(err => {
    console.error("brain geometry failed", err);
    if (fallback) {
      fallback.textContent = "Connectome geometry unavailable: " + err.message;
      fallback.classList.remove("hidden");
    }
  });

  /* interaction */
  canvas.addEventListener("pointerdown", e => {
    drag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    yaw -= (e.clientX - drag.x) * 0.006;
    pitch = Math.max(-1.2, Math.min(1.2, pitch + (e.clientY - drag.y) * 0.005));
    drag = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    dist = Math.max(1.5, Math.min(4.5, dist + e.deltaY * 0.002));
  }, { passive: false });

  raf = requestAnimationFrame(frame);
}

export function setBrain3DVisible(v) {
  visible = !!v;
  if (v) { labelTick = 1; resize(); }
}

/* activations: array of 4 base64 uint8 buffers (one per instance) */
const lastAct = [null, null, null, null];
export function setBrainActivations(b64list) {
  if (!built || !b64list) return;
  const now = performance.now();
  const tmp = new THREE.Color();
  for (let q = 0; q < 4 && q < b64list.length; q++) {
    const bytes = b64Bytes(b64list[q]);
    const n = Math.min(G.units, bytes.length);
    const tc = targetCol[q];
    const prev = lastAct[q];
    for (let i = 0; i < n; i++) {
      const act = bytes[i];
      const jumped = prev && Math.abs(act - prev[i]) >= 24;
      rampColor(act, tmp);
      tc[i * 3] = tmp.r; tc[i * 3 + 1] = tmp.g; tc[i * 3 + 2] = tmp.b;
      if (jumped) flash[q].set(i, now + 500);
      else if (!flash[q].has(i)) meshes[q].setColorAt(i, tmp);
    }
    lastAct[q] = bytes;
    meshes[q].instanceColor.needsUpdate = true;
  }
}

window.brain3D = {
  init: initBrain3D,
  setVisible: setBrain3DVisible,
  setActivations: setBrainActivations,
  get ready() { return built; },
};

/* module scripts are deferred — DOM is ready here */
initBrain3D();
