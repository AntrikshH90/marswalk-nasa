/* MARSWALK mission control
   Custom canvas slippy-map over the server tile pyramid, EVA routing,
   ML probes, live NASA feeds, 3D terrain.
   No CDN dependencies: three.js is vendored under /static/vendor. */

// ============================================================ small helpers
const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const HEX = (r, g, b) => `rgb(${r},${g},${b})`;

const MINERAL_COLORS = {
  phyllosilicate: [126, 200, 140], carbonate: [240, 214, 120], olivine: [176, 208, 74],
  pyroxene: [132, 106, 178], silica: [99, 205, 214], sulfate: [244, 156, 99],
  dune_sand: [224, 178, 120],
};
const LAYER_SWATCH = {
  terrain: "#8a5a3c", ortho: "#b08a66", themis: "#c97b4a", mola: "#d8a05a",
  hillshade: "#7a7a7a", slopeh: "#e04a3a", roughh: "#4ad0c8", mineral: "#7ec88c",
  trav: "#2ecc82", science: "#ffb02e", objective: "#ff8c42",
};

// ============================================================ application state
const S = {
  meta: null, W: 0, H: 0, pxm: 40,
  levels: [8, 4, 2, 1],
  // view: centre in WORK-grid pixels and a continuous zoom in screen px per
  // work px. The tile level is derived from the zoom, so the map can be framed
  // smoothly instead of snapping between 4 fixed scales.
  view: { cx: 0, cy: 0, zoom: 0.4 },
  base: "terrain",
  overlays: new Map(),          // id -> opacity 0..1
  waypoints: [],                // {x,y,name,id?}
  targets: [],
  selectedTarget: null,
  probe: null,
  route: null,
  hoverWork: null,
  mode: "nav",                  // 'nav' | 'plan'
  tiles: new Map(),
  pending: 0,
  conditions: null,
  roverTrack: null,             // real Perseverance driven track (ground truth)
  showRover: true,
};

// ============================================================ map tile cache
function tileKey(layer, level, tx, ty) { return `${layer}/${level}/${tx}/${ty}`; }

function getTile(layer, level, tx, ty) {
  const k = tileKey(layer, level, tx, ty);
  let t = S.tiles.get(k);
  if (t) return t;
  t = { img: new Image(), ready: false, failed: false };
  t.img.onload = () => { t.ready = true; S.pending--; scheduleDraw(); };
  t.img.onerror = () => { t.failed = true; S.pending--; scheduleDraw(); };
  t.img.src = `/api/tiles/${layer}/${level}/${tx}/${ty}.png`;
  S.pending++;
  S.tiles.set(k, t);
  return t;
}

function levelDims(level) {
  const f = S.levels[level];
  return { w: Math.floor(S.W / f), h: Math.floor(S.H / f), f };
}

/* Pick the pyramid level whose native scale is closest to the display scale.
   level scale on screen = zoom * factor, so choosing the factor nearest to
   1/zoom keeps that product inside [0.71, 1.41]: never magnified enough to
   look soft, never downscaled enough to pull in a swarm of tiles. */
function pickLevel(zoom) {
  let best = 0, bestErr = Infinity;
  for (let L = 0; L < S.levels.length; L++) {
    const err = Math.abs(Math.log2(zoom * S.levels[L]));
    if (err < bestErr) { bestErr = err; best = L; }
  }
  return best;
}

// work grid px  <->  screen px
const workToScreen = (wx, wy) => [
  (wx - S.view.cx) * S.view.zoom + VW / 2,
  (wy - S.view.cy) * S.view.zoom + VH / 2,
];
const screenToWork = (mx, my) => [
  (mx - VW / 2) / S.view.zoom + S.view.cx,
  (my - VH / 2) / S.view.zoom + S.view.cy,
];

// ============================================================ canvas plumbing
const map = $("map");
const mctx = map.getContext("2d");
let VW = 0, VH = 0, DPR = 1;

function resizeMap() {
  const r = map.parentElement.getBoundingClientRect();
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  VW = Math.max(1, Math.floor(r.width));
  VH = Math.max(1, Math.floor(r.height));
  map.width = Math.floor(VW * DPR);
  map.height = Math.floor(VH * DPR);
  map.style.width = VW + "px";
  map.style.height = VH + "px";
  scheduleDraw();
}

let drawQueued = false;
function scheduleDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}

const ZOOM_MIN_FACTOR = 0.5;   // how far past "whole map" you may zoom out
const ZOOM_MAX = 4.0;          // 4 screen px per 40 m work px

function minZoom() { return Math.min(VW / S.W, VH / S.H) * ZOOM_MIN_FACTOR; }

function clampView() {
  S.view.zoom = clamp(S.view.zoom, minZoom(), ZOOM_MAX);
  // let the centre wander a little past the edges so border features are reachable
  const slackX = (VW / (2 * S.view.zoom)) * 0.9;
  const slackY = (VH / (2 * S.view.zoom)) * 0.9;
  S.view.cx = clamp(S.view.cx, -slackX, S.W + slackX);
  S.view.cy = clamp(S.view.cy, -slackY, S.H + slackY);
}

function fitToBounds(pad = 1.04) {
  if (!S.meta) return;
  S.view.zoom = Math.min(VW / S.W, VH / S.H) / pad;
  S.view.cx = S.W / 2;
  S.view.cy = S.H / 2;
  clampView();
  scheduleDraw();
}

/** Frame a rectangle given in work-grid pixels (used to frame the route). */
function frameWorkBox(x0, y0, x1, y1, margin = 1.45) {
  const bw = Math.max(200, x1 - x0), bh = Math.max(200, y1 - y0);
  S.view.cx = (x0 + x1) / 2;
  S.view.cy = (y0 + y1) / 2;
  S.view.zoom = Math.min(VW / (bw * margin), VH / (bh * margin));
  clampView();
  scheduleDraw();
}

/** Zoom by whole steps, keeping the work-grid point under the cursor fixed. */
function zoomBy(steps, ax, ay) {
  const axx = ax == null ? VW / 2 : ax;
  const ayy = ay == null ? VH / 2 : ay;
  const before = screenToWork(axx, ayy);
  S.view.zoom = clamp(S.view.zoom * Math.pow(1.7, steps), minZoom(), ZOOM_MAX);
  const after = screenToWork(axx, ayy);
  S.view.cx += before[0] - after[0];
  S.view.cy += before[1] - after[1];
  clampView();
  scheduleDraw();
}

function flyTo(workX, workY, zoom) {
  S.view.cx = workX;
  S.view.cy = workY;
  if (zoom != null) S.view.zoom = zoom;
  clampView();
  scheduleDraw();
}

// ============================================================ draw the map
function draw() {
  if (!S.meta) return;
  mctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  mctx.clearRect(0, 0, VW, VH);
  mctx.fillStyle = "#070910";
  mctx.fillRect(0, 0, VW, VH);

  const L = pickLevel(S.view.zoom);
  const f = S.levels[L];
  const s = S.view.zoom * f;            // screen px per level px
  const d = levelDims(L);
  const cols = Math.ceil(d.w / 256), rows = Math.ceil(d.h / 256);

  const [ux0, uy0] = screenToWork(0, 0);
  const [ux1, uy1] = screenToWork(VW, VH);
  const tx0 = Math.floor((ux0 / f) / 256), tx1 = Math.floor((ux1 / f) / 256);
  const ty0 = Math.floor((uy0 / f) / 256), ty1 = Math.floor((uy1 / f) / 256);

  mctx.imageSmoothingEnabled = true;
  mctx.imageSmoothingQuality = "high";
  const drawLayer = (layer, alpha) => {
    mctx.globalAlpha = alpha;
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        if (tx < 0 || ty < 0 || tx >= cols || ty >= rows) continue;
        const t = getTile(layer, L, tx, ty);
        if (!t.ready) continue;
        const [px, py] = workToScreen(tx * 256 * f, ty * 256 * f);
        // 1 px of bleed hides hairline seams at fractional scales
        mctx.drawImage(t.img, px, py, 256 * s + 1, 256 * s + 1);
      }
    }
    mctx.globalAlpha = 1;
  };

  drawLayer(S.base, 1);
  for (const [id, op] of S.overlays) drawLayer(id, op);

  drawGraticule();
  drawRoverTrack();
  drawRoute();
  drawTargets();
  drawHab();          // last, so the home base is never buried under the route
  drawScaleBar();
}

/** The rover's actual driven track (NASA/NAIF SPICE ground truth). */
function drawRoverTrack() {
  const T = S.roverTrack;
  if (!T || !T.points || T.points.length < 2 || !S.showRover) return;
  mctx.save();
  mctx.lineJoin = "round"; mctx.lineCap = "round";
  mctx.beginPath();
  T.points.forEach(([x, y], i) => {
    const [sx, sy] = workToScreen(x, y);
    return i ? mctx.lineTo(sx, sy) : mctx.moveTo(sx, sy);
  });
  mctx.strokeStyle = "rgba(4,16,26,.85)"; mctx.lineWidth = 5; mctx.stroke();
  mctx.strokeStyle = "rgba(120,235,255,.95)"; mctx.lineWidth = 2; mctx.stroke();

  const [ax, ay] = workToScreen(T.points[0][0], T.points[0][1]);
  const [bx, by] = workToScreen(T.points[T.points.length - 1][0],
                                T.points[T.points.length - 1][1]);
  mctx.fillStyle = "rgba(120,235,255,.95)";
  mctx.beginPath(); mctx.arc(ax, ay, 4, 0, Math.PI * 2); mctx.fill();
  mctx.beginPath(); mctx.arc(bx, by, 6, 0, Math.PI * 2);
  mctx.fillStyle = "rgba(120,235,255,.35)"; mctx.fill();
  mctx.beginPath(); mctx.arc(bx, by, 3.2, 0, Math.PI * 2);
  mctx.fillStyle = "#c9f6ff"; mctx.fill();
  if (S.view.zoom > 0.55) {
    mctx.font = "bold 10px ui-monospace,monospace";
    mctx.fillStyle = "#c9f6ff";
    mctx.shadowColor = "rgba(0,0,0,.9)"; mctx.shadowBlur = 4;
    mctx.fillText(`PERSEVERANCE · sol ${T.sol_range[1]}`, bx + 9, by + 4);
    mctx.shadowBlur = 0;
  }
  mctx.restore();
}

/** The landing site / habitat. Drawn on top of the route on purpose. */
function drawHab() {
  const [hx, hy] = workToScreen(S.meta.hab.px.x, S.meta.hab.px.y);
  if (hx < -40 || hy < -40 || hx > VW + 40 || hy > VH + 40) return;
  mctx.save();
  mctx.translate(hx, hy);
  mctx.fillStyle = "#5ec8e8";
  mctx.strokeStyle = "#04121a";
  mctx.lineWidth = 2.5;
  mctx.beginPath();
  mctx.moveTo(0, -9); mctx.lineTo(9, 0); mctx.lineTo(0, 9); mctx.lineTo(-9, 0);
  mctx.closePath();
  mctx.fill(); mctx.stroke();
  mctx.fillStyle = "#eaffff";
  mctx.font = "bold 11px ui-monospace,monospace";
  mctx.shadowColor = "rgba(0,0,0,.9)"; mctx.shadowBlur = 4;
  mctx.fillText("HAB-1", 14, 4);
  mctx.restore();
}

function drawGraticule() {
  const bb = S.meta.grid.bbox;
  const degPerPx = (bb.e - bb.w) / S.W / S.view.zoom;
  const stepDeg = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    .find((v) => v >= degPerPx * 140) || 5;
  mctx.save();
  mctx.strokeStyle = "rgba(255,255,255,.10)";
  mctx.fillStyle = "rgba(255,255,255,.36)";
  mctx.lineWidth = 1;
  mctx.font = "9px ui-monospace,monospace";
  const lonToX = (lon) => workToScreen((lon - bb.w) / (bb.e - bb.w) * S.W, 0)[0];
  const latToY = (lat) => workToScreen(0, (bb.n - lat) / (bb.n - bb.s) * S.H)[1];
  for (let lon = Math.ceil(bb.w / stepDeg) * stepDeg; lon <= bb.e; lon += stepDeg) {
    const x = lonToX(lon);
    if (x < -20 || x > VW + 20) continue;
    mctx.beginPath(); mctx.moveTo(x, 0); mctx.lineTo(x, VH); mctx.stroke();
    mctx.fillText(lon.toFixed(2) + "°E", x + 3, VH - 6);
  }
  for (let lat = Math.ceil(bb.s / stepDeg) * stepDeg; lat <= bb.n; lat += stepDeg) {
    const y = latToY(lat);
    if (y < -20 || y > VH + 20) continue;
    mctx.beginPath(); mctx.moveTo(0, y); mctx.lineTo(VW, y); mctx.stroke();
    mctx.fillText(lat.toFixed(2) + "°N", 4, y - 3);
  }
  mctx.restore();
}

function drawScaleBar() {
  const mPerScreenPx = S.pxm / S.view.zoom;
  const raw = 120 * mPerScreenPx;
  const nice = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    .find((v) => v >= raw) || 500000;
  const px = nice / mPerScreenPx;
  const x = VW - px - 24, y = VH - 26;
  mctx.save();
  mctx.strokeStyle = "rgba(255,255,255,.8)";
  mctx.fillStyle = "rgba(255,255,255,.9)";
  mctx.lineWidth = 2;
  mctx.beginPath();
  mctx.moveTo(x, y - 5); mctx.lineTo(x, y); mctx.lineTo(x + px, y); mctx.lineTo(x + px, y - 5);
  mctx.stroke();
  mctx.font = "10px ui-monospace,monospace";
  const lbl = nice >= 1000 ? (nice / 1000) + " km" : nice + " m";
  const w = mctx.measureText(lbl).width;
  mctx.fillText(lbl, x + px / 2 - w / 2, y - 8);
  mctx.restore();
}

function drawTargets() {
  for (const t of S.targets) {
    const [x, y] = workToScreen(t.px, t.py);
    if (x < -30 || y < -30 || x > VW + 30 || y > VH + 30) continue;
    const sel = S.selectedTarget === t.id;
    const r = sel ? 8 : 6;
    mctx.save();
    mctx.beginPath(); mctx.arc(x, y, r, 0, Math.PI * 2);
    mctx.fillStyle = sel ? "#ffb02e" : "rgba(255,176,46,.28)";
    mctx.fill();
    mctx.lineWidth = 2;
    mctx.strokeStyle = sel ? "#fff3d6" : "#ffb02e";
    mctx.stroke();
    mctx.fillStyle = sel ? "#1a0e06" : "#ffd98a";
    mctx.font = "bold 8px ui-monospace,monospace";
    mctx.textAlign = "center"; mctx.textBaseline = "middle";
    mctx.fillText(t.id.replace("T", ""), x, y + 0.5);
    mctx.restore();
  }
}

function drawRoute() {
  if (S.route && S.route.path && S.route.path.length > 1) {
    mctx.save();
    mctx.lineJoin = "round"; mctx.lineCap = "round";
    const pts = S.route.path.map((p) => workToScreen(p.x, p.y));
    mctx.beginPath();
    pts.forEach(([x, y], i) => (i ? mctx.lineTo(x, y) : mctx.moveTo(x, y)));
    mctx.strokeStyle = "rgba(0,0,0,.55)"; mctx.lineWidth = 6; mctx.stroke();
    mctx.strokeStyle = "#33e08a"; mctx.lineWidth = 3; mctx.stroke();
    mctx.setLineDash([7, 7]);
    mctx.strokeStyle = "rgba(255,255,255,.75)"; mctx.lineWidth = 1.4; mctx.stroke();
    mctx.setLineDash([]);
    mctx.restore();
  }
  S.waypoints.forEach((w, i) => {
    const [x, y] = workToScreen(w.x, w.y);
    mctx.save();
    mctx.beginPath(); mctx.arc(x, y, 9, 0, Math.PI * 2);
    mctx.fillStyle = "rgba(255,107,53,.9)"; mctx.fill();
    mctx.strokeStyle = "#fff"; mctx.lineWidth = 2; mctx.stroke();
    mctx.fillStyle = "#1a0e06"; mctx.font = "bold 10px ui-monospace,monospace";
    mctx.textAlign = "center"; mctx.textBaseline = "middle";
    mctx.fillText(String(i + 1), x, y + 0.5);
    mctx.restore();
  });
}

// ============================================================ map interaction
let drag = null;
map.addEventListener("pointerdown", (e) => {
  if (S.mode === "plan") return;
  drag = { x: e.clientX, y: e.clientY, cx: S.view.cx, cy: S.view.cy, moved: false };
  map.classList.add("dragging");
  map.setPointerCapture(e.pointerId);
});
map.addEventListener("pointermove", (e) => {
  const r = map.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    S.view.cx = drag.cx - dx / S.view.zoom;
    S.view.cy = drag.cy - dy / S.view.zoom;
    clampView(); scheduleDraw();
  }
  scheduleProbe(mx, my);
});
const endDrag = (e) => {
  if (!drag) return;
  const wasMoved = drag.moved;
  drag = null;
  map.classList.remove("dragging");
  try { map.releasePointerCapture(e.pointerId); } catch (_) {}
  return wasMoved;
};
map.addEventListener("pointerup", endDrag);
map.addEventListener("pointercancel", endDrag);
map.addEventListener("pointerleave", () => { hideProbe(); });
map.addEventListener("click", (e) => {
  if (S.mode !== "plan") return;
  const r = map.getBoundingClientRect();
  const [wx, wy] = screenToWork(e.clientX - r.left, e.clientY - r.top);
  if (wx < 0 || wy < 0 || wx >= S.W || wy >= S.H) return;
  addWaypoint(Math.round(wx), Math.round(wy), "Map click");
});
map.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = map.getBoundingClientRect();
  zoomBy(e.deltaY < 0 ? 1 : -1, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
map.addEventListener("dblclick", (e) => {
  const r = map.getBoundingClientRect();
  zoomBy(1, e.clientX - r.left, e.clientY - r.top);
});

$("zoom-in").onclick = () => zoomBy(1);
$("zoom-out").onclick = () => zoomBy(-1);
$("zoom-fit").onclick = () => fitToBounds();
$("btn-home").onclick = () => flyTo(S.meta.hab.px.x, S.meta.hab.px.y, 1.2);
$("btn-rover").onclick = () => {
  S.showRover = !S.showRover;
  $("btn-rover").classList.toggle("active", S.showRover);
  scheduleDraw();
};

document.querySelectorAll("#map-tools .tool[data-mode]").forEach((b) => {
  b.onclick = () => {
    S.mode = b.dataset.mode;
    document.querySelectorAll("#map-tools .tool[data-mode]").forEach((o) =>
      o.classList.toggle("active", o === b));
    map.classList.toggle("planmode", S.mode === "plan");
  };
});

// keyboard
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "+" || e.key === "=") zoomBy(1);
  else if (e.key === "-" || e.key === "_") zoomBy(-1);
  else if (e.key === "f") fitToBounds();
  else if (e.key === "h") flyTo(S.meta.hab.px.x, S.meta.hab.px.y, 1.2);
  else if (e.key === "Escape" && !$("view3d").classList.contains("hidden")) close3D();
});

// ============================================================ probe
let probeTimer = null, probeSeq = 0;
function scheduleProbe(mx, my) {
  if (probeTimer) clearTimeout(probeTimer);
  probeTimer = setTimeout(() => doProbe(mx, my), 60);
}
async function doProbe(mx, my) {
  const [wx, wy] = screenToWork(mx, my);
  const px = Math.floor(wx), py = Math.floor(wy);
  if (px < 0 || py < 0 || px >= S.W || py >= S.H) return hideProbe();
  S.hoverWork = { x: px, y: py };
  const seq = ++probeSeq;
  try {
    const r = await fetch(`/api/probe?px=${px}&py=${py}`);
    if (!r.ok) return;
    const d = await r.json();
    if (seq !== probeSeq) return;
    S.probe = d;
    renderProbeHUD(d);
  } catch (_) {}
}
function hideProbe() {
  S.probe = null; S.hoverWork = null;
  $("hud-coord").textContent = "—";
  $("hud-elev").textContent = "ELEV —";
  $("hud-slope").textContent = "SLOPE —";
  $("hud-min").textContent = "UNIT —";
  $("hud-trav").textContent = "TRAVERSE —";
  $("hud-trav").className = "";
}
function travStatus(v) {
  if (v >= 0.7) return ["GO", "ok"];
  if (v >= 0.45) return ["CAUTION", "warn"];
  return ["HOLD", "bad"];
}
function renderProbeHUD(d) {
  $("hud-coord").textContent = `${fmt(d.lat, 4)}°N ${fmt(d.lon, 4)}°E`;
  $("hud-elev").innerHTML = `ELEV <b>${fmt(d.elev, 0)} m</b>`;
  const sc = d.slope > 20 ? "bad" : d.slope > 12 ? "warn" : "";
  $("hud-slope").className = sc;
  $("hud-slope").innerHTML = `SLOPE <b>${fmt(d.slope, 1)}°</b>`;
  $("hud-min").innerHTML = `UNIT <b>${d.mineral}</b>`;
  const [lbl, cls] = travStatus(d.traversability);
  $("hud-trav").className = cls;
  $("hud-trav").innerHTML = `<b>${lbl}</b> ${(d.traversability * 100).toFixed(0)}%`;
}

// ============================================================ layers UI
function buildLayerRack() {
  const baseBox = $("base-layers"), ovBox = $("overlay-layers");
  baseBox.innerHTML = ""; ovBox.innerHTML = "";
  for (const L of S.meta.layers) {
    const row = el("div", "layer");
    const dot = el("span", "dot");
    dot.style.background = LAYER_SWATCH[L.id] || "#888";
    const nm = el("span", "nm", L.name);
    row.title = `${L.desc}\nSource: ${L.source}`;
    row.appendChild(dot); row.appendChild(nm);
    if (L.kind === "base") {
      row.classList.toggle("active", L.id === S.base);
      row.onclick = () => {
        S.base = L.id;
        updateChip(); buildLayerRack(); scheduleDraw();
      };
      baseBox.appendChild(row);
    } else {
      const cb = el("input");
      cb.type = "checkbox";
      cb.checked = S.overlays.has(L.id);
      cb.onclick = (e) => e.stopPropagation();
      cb.onchange = () => {
        if (cb.checked) S.overlays.set(L.id, Number($("overlay-opacity").value) / 100);
        else S.overlays.delete(L.id);
        row.classList.toggle("active", cb.checked);
        renderLegend(); updateChip(); scheduleDraw();
      };
      row.appendChild(cb);
      row.classList.toggle("active", S.overlays.has(L.id));
      row.onclick = () => { cb.checked = !cb.checked; cb.onchange(); };
      ovBox.appendChild(row);
    }
  }
}
$("overlay-opacity").oninput = () => {
  const v = Number($("overlay-opacity").value) / 100;
  for (const k of S.overlays.keys()) S.overlays.set(k, v);
  scheduleDraw();
};
function updateChip() {
  const names = [S.base, ...S.overlays.keys()].map((id) => {
    const L = S.meta.layers.find((x) => x.id === id);
    return L ? `<b>${L.name}</b>` : id;
  });
  $("layer-chip").innerHTML = names.join(" + ");
}
function renderLegend() {
  const box = $("legend");
  box.innerHTML = "";
  if (S.overlays.has("mineral")) {
    for (const [k, c] of Object.entries(MINERAL_COLORS)) {
      const d = el("div", "lg");
      const sw = el("span", "sw");
      sw.style.background = HEX(...c);
      d.appendChild(sw); d.appendChild(document.createTextNode(k));
      box.appendChild(d);
    }
  }
  for (const id of S.overlays.keys()) {
    const L = S.meta.layers.find((x) => x.id === id);
    if (!L || id === "mineral") continue;
    const d = el("div", "lg", L.desc);
    d.style.width = "100%";
    box.appendChild(d);
  }
}

// ============================================================ targets
function buildTargets() {
  const box = $("target-list");
  box.innerHTML = "";
  const list = [...S.targets].sort((a, b) => b.objective - a.objective);
  list.forEach((t) => {
    const c = el("div", "card");
    c.dataset.id = t.id;
    c.innerHTML = `
      <div class="hd">
        <span class="tid">${t.id}</span>
        <span class="tnm">${t.name}</span>
      </div>
      <div class="rat">${t.rationale}</div>
      <div class="mrow">
        <span class="kv sci">SCI <b>${fmt(t.science, 2)}</b></span>
        <span class="kv">TRAV <b>${(t.traversability * 100).toFixed(0)}%</b></span>
        <span class="kv">OBJ <b>${fmt(t.objective, 2)}</b></span>
        <span class="kv">${t.mineral}</span>
        <span class="kv">${fmt(t.elev, 0)} m</span>
        <span class="kv">slope ${fmt(t.slope, 1)}°</span>
      </div>
      <button class="add">＋ ADD</button>`;
    c.onclick = (e) => {
      if (e.target.classList.contains("add")) e.stopPropagation();
      selectTarget(t.id, e.target.classList.contains("add"));
    };
    c.querySelector(".add").onclick = (e) => {
      e.stopPropagation();
      addTargetToRoute(t);
    };
    box.appendChild(c);
  });
}
function selectTarget(id, addToRoute) {
  S.selectedTarget = id;
  const t = S.targets.find((x) => x.id === id);
  if (!t) return;
  document.querySelectorAll(".card").forEach((c) =>
    c.classList.toggle("sel", c.dataset.id === id));
  renderInspector(t);
  flyTo(t.px, t.py, Math.max(S.view.zoom, 1.2));
  if (addToRoute) addTargetToRoute(t);
  else showTab("inspector");
}
function addTargetToRoute(t) {
  S.waypoints.push({ x: t.px, y: t.py, name: `${t.id} · ${t.name}`, id: t.id });
  renderLegs();
  showToast(`${t.id} added to route — press PLAN EVA ROUTE`, "good");
}
function renderLegs() {
  const ol = $("legs");
  ol.innerHTML = "";
  S.waypoints.forEach((w, i) => {
    const li = el("li");
    li.innerHTML = `<span class="idx">${i + 1}</span>
      <span class="nm">${w.name || "Waypoint"}</span>
      <span class="co">${w.x},${w.y}</span>`;
    const rm = el("button", "rm", "✕");
    rm.title = "Remove";
    rm.onclick = (e) => {
      e.stopPropagation();
      S.waypoints.splice(i, 1);
      renderLegs();
    };
    li.appendChild(rm);
    ol.appendChild(li);
  });
}
$("btn-clear").onclick = () => {
  S.waypoints = []; S.route = null;
  renderLegs(); renderStats(null); drawProfile(null); scheduleDraw();
};

// ============================================================ inspector
function renderInspector(t) {
  const b = $("inspector-body");
  const bar = (label, v, max, color) => `
    <div class="bar-row">
      <span class="bl">${label}</span>
      <span class="bt"><span class="bf" style="width:${clamp(v / max * 100, 0, 100)}%;background:${color}"></span></span>
      <span class="bv">${fmt(v, 2)}</span>
    </div>`;
  b.className = "";
  b.innerHTML = `
    <div class="insp-hd">
      <span class="tid">${t.id}</span>
      <span class="inm">${t.name}</span>
    </div>
    <div class="insp-sec">
      <h3>WHY THIS TARGET</h3>
      <div style="font-size:11px;line-height:1.6;color:var(--ink2)">${t.rationale}</div>
    </div>
    <div class="insp-sec">
      <h3>ML SCORES</h3>
      ${bar("science", t.science, 6, "linear-gradient(90deg,#5ec8e8,#ffb02e)")}
      ${bar("traversability", t.traversability, 1, "linear-gradient(90deg,#e04a3a,#2ecc82)")}
      ${bar("objective", t.objective, 6, "linear-gradient(90deg,#7ec88c,#ffb02e)")}
    </div>
    <div class="insp-sec">
      <h3>TERRAIN</h3>
      <div class="insp-grid">
        <div class="stat-box"><div class="sv">${fmt(t.elev, 0)}</div><div class="sl">ELEV m</div></div>
        <div class="stat-box"><div class="sv">${fmt(t.slope, 1)}°</div><div class="sl">SLOPE</div></div>
        <div class="stat-box"><div class="sv" style="font-size:11px">${t.mineral}</div><div class="sl">UNIT</div></div>
        <div class="stat-box"><div class="sv" style="font-size:11px">${fmt(t.lat, 3)}, ${fmt(t.lon, 3)}</div><div class="sl">LAT, LON</div></div>
      </div>
    </div>
    <button class="btn go" id="insp-add">＋ ADD TO EVA ROUTE</button>`;
  $("insp-add").onclick = () => addTargetToRoute(t);
}

// ============================================================ routing
async function planRoute() {
  if (S.waypoints.length < 2) {
    showToast("Add at least two waypoints (or two science targets).", "bad");
    return;
  }
  $("map-loading").classList.remove("hidden");
  $("map-loading").lastElementChild.textContent = "PLANNING EVA ROUTE…";
  try {
    const body = {
      waypoints: S.waypoints.map((w) => ({ x: w.x, y: w.y })),
      max_slope: Number($("max-slope").value),
      crew: Number($("crew-n").textContent),
      prefer_science: $("prefer-science").checked,
    };
    const r = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) {
      showToast(d.detail || "Route planning failed.", "bad");
      S.route = null;
    } else {
      S.route = d;
      renderStats(d.stats);
      drawProfile(d.stats);
      scheduleDraw();
      const w = d.stats.warnings || [];
      showToast(
        `Route: ${d.stats.dist_km} km · ${d.stats.time_h} h` +
        (w.length ? ` · ${w.length} advisor${w.length > 1 ? "ies" : "y"}` : " · no advisories"),
        w.length ? "bad" : "good");
    }
  } catch (e) {
    showToast("Route request failed: " + e.message, "bad");
  } finally {
    $("map-loading").classList.add("hidden");
  }
}
$("btn-plan").onclick = planRoute;

function renderStats(st) {
  const set = (id, v) => { $(id).textContent = v; };
  if (!st) {
    ["st-dist", "st-time", "st-ascent", "st-grade", "st-trav", "st-o2", "st-water", "st-power"]
      .forEach((i) => set(i, "—"));
    set("st-warn", "—");
    $("st-warn").parentElement.classList.remove("has");
    return;
  }
  set("st-dist", fmt(st.dist_km, 2));
  set("st-time", fmt(st.time_h, 2));
  set("st-ascent", fmt(st.ascent_m, 0));
  set("st-grade", fmt(st.max_grade, 1) + "°");
  set("st-trav", (st.traversability * 100).toFixed(0) + "%");
  set("st-o2", fmt(st.o2_kg, 2));
  set("st-water", fmt(st.water_l, 1));
  set("st-power", fmt(st.power_kwh, 2));
  const n = (st.warnings || []).length;
  set("st-warn", n);
  $("st-warn").parentElement.classList.toggle("has", n > 0);
  $("st-warn").title = (st.warnings || []).join("\n");
}

// ============================================================ profile chart
const prof = $("profile");
function drawProfile(st) {
  const ctx = prof.getContext("2d");
  const r = prof.parentElement.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(1, Math.floor(prof.parentElement.clientWidth));
  const H = Math.max(1, Math.floor(prof.clientHeight || (r.height - 78)));
  prof.width = W * dpr; prof.height = H * dpr;
  prof.style.width = W + "px"; prof.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!st || !st.profile || !st.profile.dist_km || st.profile.dist_km.length < 2) {
    ctx.fillStyle = "rgba(255,255,255,.22)";
    ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("ELEVATION PROFILE — plan a route to populate", 14, H / 2);
    return;
  }
  const P = st.profile;
  const dmax = P.dist_km[P.dist_km.length - 1] || 1;
  const zmin = Math.min(...P.elev_m), zmax = Math.max(...P.elev_m);
  const zpad = Math.max(20, (zmax - zmin) * 0.12);
  const z0 = zmin - zpad, z1 = zmax + zpad;
  const L = 52, R = 14, T = 16, B = 22;
  const pw = W - L - R, ph = H - T - B;
  const X = (d) => L + (d / dmax) * pw;
  const Y = (z) => T + (1 - (z - z0) / (z1 - z0)) * ph;

  // grid
  ctx.strokeStyle = "rgba(255,255,255,.07)";
  ctx.fillStyle = "rgba(255,255,255,.35)";
  ctx.font = "9px ui-monospace,monospace";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const z = z0 + (z1 - z0) * (i / 4);
    const y = Y(z);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillText(Math.round(z) + " m", 6, y + 3);
  }
  const xt = Math.max(1, Math.round(dmax / 6));
  for (let d = 0; d <= dmax; d += dmax / 6) {
    const x = X(d);
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke();
    ctx.fillText(d.toFixed(1) + " km", x - 12, H - 6);
  }

  // filled area with grade colouring
  const grad = ctx.createLinearGradient(0, T, 0, H - B);
  grad.addColorStop(0, "rgba(255,176,46,.46)");
  grad.addColorStop(1, "rgba(193,68,14,.06)");
  ctx.beginPath();
  ctx.moveTo(X(P.dist_km[0]), H - B);
  for (let i = 0; i < P.dist_km.length; i++) ctx.lineTo(X(P.dist_km[i]), Y(P.elev_m[i]));
  ctx.lineTo(X(P.dist_km[P.dist_km.length - 1]), H - B);
  ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

  // grade-coloured top line
  ctx.lineWidth = 2;
  for (let i = 1; i < P.dist_km.length; i++) {
    const g = Math.abs(P.grade[i] || 0);
    ctx.strokeStyle = g > 18 ? "#ff4d5e" : g > 12 ? "#ffcc44" : g > 6 ? "#8ae08a" : "#33e08a";
    ctx.beginPath();
    ctx.moveTo(X(P.dist_km[i - 1]), Y(P.elev_m[i - 1]));
    ctx.lineTo(X(P.dist_km[i]), Y(P.elev_m[i]));
    ctx.stroke();
  }

  // leg markers
  (st.stops_km || []).forEach((d, i) => {
    const x = X(d);
    ctx.strokeStyle = "rgba(255,107,53,.85)";
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff6b35";
    ctx.beginPath(); ctx.arc(x, T, 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#1a0e06";
    ctx.font = "bold 9px ui-monospace,monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), x, T + 0.5);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  });
}

// ============================================================ conditions
async function loadConditions() {
  try {
    const r = await fetch("/api/live/conditions");
    if (!r.ok) throw new Error("http " + r.status);
    const d = await r.json();
    S.conditions = d;
    renderConditions(d);
    renderFlares(d.space_weather);
  } catch (e) {
    $("advisory").textContent = "Conditions feed unavailable: " + e.message;
  }
}
function renderConditions(d) {
  const dust = d.dust || {}, rad = d.radiation || {};
  $("cond-origin").textContent = d.origin || "—";
  $("cond-origin").className = "badge " + (d.origin === "LIVE" ? "live" : "cached");

  $("dust-tau").textContent = dust.season_probability != null
    ? (dust.season_probability * 100).toFixed(0) + "%"
    : "—";
  $("dust-bar").style.width = clamp((dust.season_probability || 0) * 100, 3, 100) + "%";
  $("dust-risk").textContent = dust.risk || "—";
  $("dust-risk").style.color = dust.risk === "HIGH" ? "var(--bad)"
    : dust.risk === "MODERATE" ? "var(--warn)" : "var(--ok)";

  $("rad-level").textContent = rad.level || "—";
  $("rad-level").style.color = rad.level === "WARN" ? "var(--bad)"
    : rad.level === "CAUTION" ? "var(--warn)" : "var(--ok)";
  $("rad-bar").style.width = clamp(rad.score || 6, 4, 100) + "%";
  $("rad-sub").textContent = rad.sep_72h
    ? `${rad.sep_72h} SEP event(s) / 72 h`
    : (rad.largest_flare_48h ? `flare ${rad.largest_flare_48h} / 48 h` : "no SEP events");

  const adv = $("advisory");
  adv.className = "advisory " + (rad.level === "WARN" ? "warn"
    : rad.level === "CAUTION" ? "caution" : "go");
  const meas = d.measured;
  let extra = "";
  if (meas) {
    extra = `<div style="margin-top:7px;padding-top:6px;border-top:1px dashed #242b39;
      font-size:10px;color:var(--ink3)">
      Measured at ${meas.basis}: min ${fmt(meas.min_temp_c, 0)} °C,
      max ${fmt(meas.max_temp_c, 0)} °C, P ${fmt(meas.pressure_pa, 0)} Pa
      (n=${meas.n_sols} sols)</div>`;
  }
  adv.innerHTML = `<b>${dust.text || ""}</b>${rad.text ? "<br>" + rad.text : ""}${extra}`;

  const b = $("eva-go");
  b.className = "eva-badge " + (d.eva_go ? "go" : "hold");
  b.textContent = d.eva_go ? "EVA GREEN — GO" : `EVA HOLD — ${d.hold_reason || "conditions"}`;
}
function renderFlares(sw) {
  const box = $("flare-box");
  box.innerHTML = "";
  if (!sw) return;
  const head = el("div", "lbl");
  head.innerHTML = `<span>SPACE WEATHER · ${sw.origin || "—"}</span>
    <span>${sw.flares_30d} FLR · ${sw.seps_30d} SEP · ${sw.cmes_10d} CME</span>`;
  box.appendChild(head);

  (sw.recent_sep || []).slice(0, 3).forEach((s) => {
    const d = el("div", "flare SEP");
    d.innerHTML = `<span class="fc">SEP</span>
      <span class="ft">${(s.instruments || []).join(", ") || "solar energetic particles"}</span>
      <span class="fd">${(s.event_time || "").slice(0, 16).replace("T", " ")}</span>`;
    box.appendChild(d);
  });
  (sw.recent || []).slice(0, 6).forEach((f) => {
    const cls = (f.class_type || "?")[0].toUpperCase();
    const d = el("div", "flare " + (["X", "M", "C"].includes(cls) ? cls : "C"));
    d.innerHTML = `<span class="fc">${f.class_type || "—"}</span>
      <span class="ft">${f.source || "active region"}</span>
      <span class="fd">${(f.begin || "").slice(0, 16).replace("T", " ")}</span>`;
    d.title = f.note || "";
    box.appendChild(d);
  });
  if (!(sw.recent || []).length && !(sw.recent_sep || []).length) {
    box.appendChild(el("div", "empty", "No flares or SEP events reported in the last 30 days."));
  }
}

async function loadPhotos() {
  const grid = $("photo-grid");
  try {
    const r = await fetch("/api/live/photos");
    const d = await r.json();
    grid.innerHTML = "";
    if (!d.photos || !d.photos.length) {
      grid.appendChild(el("div", "empty", "No imagery available."));
      return;
    }
    d.photos.forEach((p) => {
      const c = el("div", "ph");
      const img = el("img");
      img.loading = "lazy";
      img.src = p.url;
      // Some NASA CDN assets refuse requests from certain networks. Swap in a
      // bundled real image rather than leaving a broken tile on the screen.
      const pool = d.fallbacks || [];
      let fbIndex = -1;
      img.onerror = () => {
        fbIndex++;
        if (fbIndex < pool.length) {
          img.src = pool[fbIndex];
          return;
        }
        c.remove();
      };
      c.appendChild(img);
      c.appendChild(el("div", "pt", p.title || "Mars image"));
      c.appendChild(el("div", "pc", `${p.date || ""} · ${p.center || "NASA"}`));
      if (p.nasa_id) {
        c.title = "Open in NASA Image Library";
        c.onclick = () => window.open(
          `https://images.nasa.gov/details/${encodeURIComponent(p.nasa_id)}`, "_blank");
      }
      grid.appendChild(c);
    });
    const note = document.querySelector("#pane-live .pane-note");
    if (note) note.innerHTML = `Surface imagery via <b>NASA Image &amp; Video Library</b> ` +
      `(origin: ${d.origin}) · space weather via <b>DONKI</b> (FLR / SEP / CME / GST).`;
  } catch (e) {
    grid.innerHTML = "";
    grid.appendChild(el("div", "empty", "Imagery feed unavailable: " + e.message));
  }
}

// ============================================================ tabs
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tabpane").forEach((p) =>
    p.classList.toggle("active", p.id === "pane-" + name));
}
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => showTab(t.dataset.tab);
});

// ============================================================ toast
let toastTimer = null;
function showToast(msg, kind) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast " + (kind || "");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 4200);
}

// ============================================================ 3D terrain
let three3d = null;
function loadThree() {
  return new Promise((resolve, reject) => {
    if (window.THREE) return resolve(window.THREE);
    const s = document.createElement("script");
    s.src = "/static/vendor/three.min.js";
    s.onload = () => resolve(window.THREE);
    s.onerror = () => reject(new Error("three.js failed to load"));
    document.head.appendChild(s);
  });
}

async function open3D() {
  const box = $("view3d");
  box.classList.remove("hidden");
  $("v3-info").textContent = "loading terrain…";
  try {
    const THREE = await loadThree();
    await build3D(THREE);
  } catch (e) {
    $("v3-info").textContent = "3D unavailable: " + e.message;
  }
}
function close3D() {
  $("view3d").classList.add("hidden");
  if (three3d && three3d.dispose) { try { three3d.dispose(); } catch (_) {} }
  three3d = null;
}
$("btn-3d").onclick = open3D;
$("v3-close").onclick = close3D;

async function build3D(THREE) {
  const canvas = $("c3d");
  if (three3d && three3d.dispose) { try { three3d.dispose(); } catch (_) {} }
  const W = S.W, H = S.H;
  const buf = await (await fetch("/api/elev/raw")).arrayBuffer();
  const raw = new Float32Array(buf);
  const stride = 4;                       // 40 m grid -> ~160 m mesh
  const gw = Math.floor(W / stride), gh = Math.floor(H / stride);
  const zmin = S.meta.elev_range[0], zmax = S.meta.elev_range[1];
  const zspan = Math.max(1, zmax - zmin);

  const geo = new THREE.PlaneGeometry(gw, gh, gw - 1, gh - 1);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const c = new THREE.Color();
  for (let j = 0; j < gh; j++) {
    for (let i = 0; i < gw; i++) {
      const idx = j * gw + i;
      const z = raw[(j * stride) * W + (i * stride)];
      const t = clamp((z - zmin) / zspan, 0, 1);
      pos.setZ(idx, t * gw * 0.16);          // vertical exaggeration
      c.setHSL(0.08 - 0.06 * t, 0.55, 0.20 + 0.42 * t);
      colors[idx * 3] = c.r; colors[idx * 3 + 1] = c.g; colors[idx * 3 + 2] = c.b;
    }
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();

  const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
    vertexColors: true, flatShading: false, side: THREE.DoubleSide,
  }));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070c);
  scene.add(mesh);
  scene.add(new THREE.AmbientLight(0xffffff, 0.42));
  const dir = new THREE.DirectionalLight(0xffd9b0, 1.05);
  dir.position.set(-1, 1, 1.2);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.22);
  fill.position.set(1, -0.4, -0.8);
  scene.add(fill);

  // route + targets drawn as overlays on the mesh plane
  const to3D = (wx, wy) => {
    const t = clamp((raw[Math.floor(wy) * W + Math.floor(wx)] - zmin) / zspan, 0, 1);
    return new THREE.Vector3(wx / W * gw - gw / 2, -(wy / H * gh - gh / 2),
      t * gw * 0.16 + 0.6);
  };
  if (S.route && S.route.path && S.route.path.length > 1) {
    const pts = S.route.path.map((p) => to3D(p.x, p.y));
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x33e08a })));
  }
  const mkMarker = (wx, wy, color, size) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(size, 12, 12),
      new THREE.MeshBasicMaterial({ color }));
    m.position.copy(to3D(wx, wy));
    m.position.z += size;
    scene.add(m);
  };
  S.targets.forEach((t) => mkMarker(t.px, t.py, 0xffb02e, 1.1));
  mkMarker(S.meta.hab.px.x, S.meta.hab.px.y, 0x5ec8e8, 1.6);

  const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 20000);
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true,
    // Keep the drawing buffer so the view survives a canvas read-back: browser
    // screenshot tools and canvas.toDataURL() both return an empty frame from a
    // WebGL canvas otherwise, which would make the 3D view un-capturable for a
    // slide deck.
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const st = { yaw: -0.6, pitch: 0.95, dist: gw * 1.25, tx: 0, ty: 0 };
  let dragging = false, panning = false, lx = 0, ly = 0;
  canvas.onpointerdown = (e) => {
    dragging = true; panning = e.button === 2 || e.shiftKey;
    lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId);
  };
  canvas.onpointerup = (e) => { dragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch (_) {} };
  canvas.onpointermove = (e) => {
    if (!dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    if (panning) { st.tx -= dx * st.dist / 900; st.ty += dy * st.dist / 900; }
    else { st.yaw -= dx * 0.006; st.pitch = clamp(st.pitch + dy * 0.005, 0.12, 1.45); }
  };
  canvas.onwheel = (e) => {
    e.preventDefault();
    st.dist = clamp(st.dist * (e.deltaY > 0 ? 1.12 : 0.89), gw * 0.25, gw * 4);
  };
  canvas.oncontextmenu = (e) => e.preventDefault();
  let down = false;
  canvas.addEventListener("pointerdown", () => { down = true; });
  window.addEventListener("pointerup", () => { down = false; });

  function frame() {
    if (!three3d || three3d.closed) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    }
    const cy = Math.cos(st.pitch), sy = Math.sin(st.pitch);
    camera.position.set(
      st.tx + st.dist * cy * Math.sin(st.yaw),
      st.ty + st.dist * cy * Math.cos(st.yaw),
      st.dist * sy);
    camera.up.set(0, 0, 1);
    camera.lookAt(st.tx, st.ty, 0);
    renderer.render(scene, camera);
    three3d.raf = requestAnimationFrame(frame);
  }
  three3d = {
    closed: false,
    dispose() {
      this.closed = true;
      cancelAnimationFrame(this.raf);
      geo.dispose();
      renderer.dispose();
    },
  };
  $("v3-info").textContent =
    `· ${gw}×${gh} mesh · ${S.route ? "route + " : ""}${S.targets.length} targets`;
  frame();
  window.setTimeout(() => {
    const t = renderer.info.render;
    if (t && t.triangles) {
      $("v3-info").textContent += ` · ${(t.triangles / 1000).toFixed(0)}k tris drawn`;
    }
  }, 1200);
}

// ============================================================ export
$("btn-export").onclick = () => {
  const payload = {
    product: "MARSWALK EVA PLAN",
    generated_utc: new Date().toISOString(),
    site: {
      name: S.meta.name,
      grid: S.meta.grid,
      elevation_range_m: S.meta.elev_range,
      hab: S.meta.hab,
    },
    mars_clock: S.meta.mars,
    eva: {
      crew: Number($("crew-n").textContent),
      max_slope_deg: Number($("max-slope").value),
      prefer_science: $("prefer-science").checked,
      waypoints: S.waypoints.map((w, i) => ({ leg: i + 1, name: w.name, px: w.x, py: w.y })),
    },
    route: S.route ? {
      stats: S.route.stats,
      science_targets_encountered: S.route.encountered,
      path_px: S.route.path,
    } : null,
    conditions: S.conditions,
    models: S.meta.metrics,
    data_sources: S.meta.sources,
    caveat: "Planning estimates for challenge/analog use — not certified flight operations.",
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `marswalk_eva_plan_sol${S.meta.mars.sol}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
  showToast("EVA plan exported as JSON.", "good");
};

// ============================================================ EVA params
document.querySelectorAll(".stepper button").forEach((b) => {
  b.onclick = () => {
    const n = clamp(Number($("crew-n").textContent) + Number(b.dataset.step), 1, 6);
    $("crew-n").textContent = n;
    if (S.route) planRoute();
  };
});
$("max-slope").oninput = () => { $("slope-v").textContent = $("max-slope").value + "°"; };
$("max-slope").onchange = () => { if (S.route) planRoute(); };
$("prefer-science").onchange = () => { if (S.route) planRoute(); };

// ============================================================ clock
function renderClock(m) {
  $("hud-sol").innerHTML = `SOL <b>${m.sol}</b>`;
  $("hud-ls").innerHTML = `Ls <b>${fmt(m.ls_deg, 2)}°</b>`;
  $("hud-lmst").innerHTML = `LMST <b>${m.lmst}</b>`;
  $("hud-utc").textContent = m.now_utc.slice(11, 19) + " UTC";
  $("hud-sol").title = `Mars Year ${m.my} · ${m.season} · sol fraction ${m.sol_fraction}`;
}
async function refreshClock() {
  try {
    const r = await fetch("/api/clock");
    if (!r.ok) return;
    const m = await r.json();
    S.meta.mars = m;
    renderClock(m);
  } catch (_) {}
}

/** Frame the planned route (plus its waypoints) with a margin. */
function frameRoute(margin = 1.5) {  if (!S.route || !S.route.path || S.route.path.length < 2) return fitToBounds();
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of S.route.path) {
    if (p.x < x0) x0 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.x > x1) x1 = p.x;
    if (p.y > y1) y1 = p.y;
  }
  for (const w of S.waypoints) {
    x0 = Math.min(x0, w.x); y0 = Math.min(y0, w.y);
    x1 = Math.max(x1, w.x); y1 = Math.max(y1, w.y);
  }
  frameWorkBox(x0, y0, x1, y1, margin);
}

/** Load the rover's real driven track (NASA/NAIF SPICE ground truth). */
async function loadRoverTrack() {
  try {
    const r = await fetch("/api/rover/traverse");
    if (!r.ok) throw new Error("http " + r.status);
    const d = await r.json();
    S.roverTrack = {
      points: (d.points || []).map((p) => [p.x, p.y]),
      sol_range: d.sol_range || [0, 0],
      distance_km: d.distance_km,
      coverage_utc: d.coverage_utc || [],
    };
    const note = document.getElementById("rover-note");
    if (note) {
      note.innerHTML =
        `<b>PERSEVERANCE (real)</b> sols ${S.roverTrack.sol_range[0]}–` +
        `${S.roverTrack.sol_range[1]} · ${S.roverTrack.distance_km} km driven · ` +
        `NASA/NAIF SPICE ground truth`;
    }
    scheduleDraw();
  } catch (e) {
    showToast("Rover traverse unavailable: " + e.message, "bad");
  }
}

// ============================================================ boot
async function boot() {
  try {
    const r = await fetch("/api/meta");
    S.meta = await r.json();
  } catch (e) {
    showToast("Cannot reach the MARSWALK API: " + e.message, "bad");
    return;
  }
  S.W = S.meta.grid.W; S.H = S.meta.grid.H; S.pxm = S.meta.grid.px_m;
  S.levels = S.meta.levels || [8, 4, 2, 1];
  S.targets = S.meta.targets || [];

  resizeMap();
  buildLayerRack();
  updateChip();
  buildTargets();
  renderLegs();
  renderClock(S.meta.mars);

  const src = $("sources");
  src.innerHTML = "<b>DATA:</b> " + (S.meta.sources || []).join(" · ") +
    "  |  <b>ML:</b> " +
    `traversability ${(S.meta.metrics.traversability_clean_acc * 100).toFixed(1)}% vs EVA rule · ` +
    `science R² ${S.meta.metrics.science_r2}`;

  fitToBounds();

  // demo route: HAB -> two highest-objective targets
  const top = [...S.targets].sort((a, b) => b.objective - a.objective).slice(0, 2);
  S.waypoints = [{ x: S.meta.hab.px.x, y: S.meta.hab.px.y, name: "HAB-1 · Octavia E. Butler Landing" },
    ...top.map((t) => ({ x: t.px, y: t.py, name: `${t.id} · ${t.name}`, id: t.id }))];
  renderLegs();
  await planRoute();
  frameRoute();

  $("map-loading").classList.add("hidden");
  $("hud-live").className = "pill ok";
  $("hud-live").textContent = "◉ LINK OK";

  loadConditions();
  loadPhotos();
  loadRoverTrack();
  setInterval(refreshClock, 30000);
  setInterval(loadConditions, 300000);
}

window.addEventListener("resize", () => { resizeMap(); drawProfile(S.route && S.route.stats); });
boot();
