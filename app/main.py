"""MARSWALK - FastAPI backend: layered tile service, EVA route planner,
ML probes, live NASA feeds."""
import io
import json
import math
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import mars_time, nasa_live  # noqa: E402
from app.routing import RoutePlanner, stats_for_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS_DIR = os.path.join(ROOT, "layers")
MODELS_DIR = os.path.join(ROOT, "models")
STATIC = os.path.join(ROOT, "static")

app = FastAPI(title="MARSWALK API", version="2.0")

# ----------------------------------------------------------------- state
meta = json.load(open(os.path.join(LAYERS_DIR, "meta.json")))
targets_doc = json.load(open(os.path.join(ROOT, "targets.json")))
metrics = json.load(open(os.path.join(MODELS_DIR, "metrics.json")))
verification_path = os.path.join(MODELS_DIR, "verification.json")
verification = (json.load(open(verification_path))
                if os.path.exists(verification_path) else {})
W, H = meta["W"], meta["H"]
BBOX = meta["bbox"]
JEZERO_LON = (BBOX["w"] + BBOX["e"]) / 2.0

# Grids the API depends on. A missing one is a build problem, so say so
# clearly at import time rather than serving a half-dead map.
REQUIRED_GRIDS = ["elev", "slope", "rough", "aspect", "mineral_idx",
                  "mineral_conf", "trav", "science", "objective"]
_missing = [n for n in REQUIRED_GRIDS
            if not os.path.exists(os.path.join(LAYERS_DIR, f"{n}.npy"))]
if _missing:
    raise RuntimeError(
        "missing derived layers: " + ", ".join(_missing) +
        "\nRun:  python scripts/preprocess.py && python scripts/train_ml.py")

_elev = np.load(os.path.join(LAYERS_DIR, "elev.npy"))
_slope = np.load(os.path.join(LAYERS_DIR, "slope.npy"))
_rough = np.load(os.path.join(LAYERS_DIR, "rough.npy"))
_aspect = np.load(os.path.join(LAYERS_DIR, "aspect.npy"))
_mid = np.load(os.path.join(LAYERS_DIR, "mineral_idx.npy"))
_mconf = np.load(os.path.join(LAYERS_DIR, "mineral_conf.npy"))
_trav = np.load(os.path.join(LAYERS_DIR, "trav.npy"))
_sci = np.load(os.path.join(LAYERS_DIR, "science.npy"))
_obj = np.load(os.path.join(LAYERS_DIR, "objective.npy"))

RGBA_FILES = {  # on-disk RGBA rasters
    "terrain": "terrain.png", "hillshade": "hillshade.png", "slopeh": "slope.png",
    "roughh": "rough.png", "mineral": "mineral.png", "ortho": "ortho.png",
    "themis": "themis.png", "mola": "mola.png",
}
SCALAR_CMAPS = {
    "trav": (_trav, "trav"),
    "science": (_sci, "science"),
    "objective": (_obj, "obj"),
}
LEVEL_FACTORS = [8, 4, 2, 1]  # level -> downsample factor
TILE = 256


# ------------------------------------------------------------ colormaps
def _cmap_arr(stops):
    """stops: list of (t, r, g, b). returns f(x in 0..1) -> rgb array."""
    arr = np.array(stops, np.float32)

    def f(x):
        x = np.clip(x, 0, 1)
        out = np.zeros(x.shape + (3,), np.float32)
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i + 1]
            m = (x >= a[0]) & (x <= b[0])
            t = np.zeros_like(x)
            t[m] = (x[m] - a[0]) / max(b[0] - a[0], 1e-9)
            for c in range(3):
                out[..., c][m] = a[1 + c] + t[m] * (b[1 + c] - a[1 + c])
        return out.astype(np.uint8)

    return f


_CMAPS = {
    "trav": _cmap_arr([(0.0, 168, 42, 42), (0.35, 224, 120, 40), (0.6, 230, 200, 70),
                       (0.8, 120, 210, 120), (1.0, 30, 200, 120)]),
    "science": _cmap_arr([(0.0, 14, 20, 40), (0.3, 40, 90, 160), (0.55, 60, 180, 170),
                          (0.75, 170, 210, 70), (1.0, 255, 230, 90)]),
    "obj": _cmap_arr([(0.0, 10, 14, 28), (0.3, 70, 50, 140), (0.55, 40, 150, 140),
                      (0.8, 120, 200, 80), (1.0, 255, 210, 60)]),
}

_pyr_cache = {}   # layer -> {level: rgba ndarray}
_rgba_cache = {}  # layer -> full rgba


def get_rgba(layer):
    if layer in _rgba_cache:
        return _rgba_cache[layer]
    if layer in RGBA_FILES:
        path = os.path.join(LAYERS_DIR, RGBA_FILES[layer])
        if not os.path.exists(path):
            return None
        arr = np.array(Image.open(path).convert("RGBA"))
    elif layer in SCALAR_CMAPS:
        data, cmap_name = SCALAR_CMAPS[layer]
        rgb = _CMAPS[cmap_name](data / (data.max() + 1e-9))
        dmin, dmax = float(data.min()), float(data.max())
        a = np.where(data > (dmin + 0.02 * (dmax - dmin)), 190, 0).astype(np.uint8)
        arr = np.dstack([rgb, a])
    else:
        return None
    _rgba_cache[layer] = arr
    return arr


def get_level_rgba(layer, level):
    key = (layer, level)
    if key in _pyr_cache:
        return _pyr_cache[key]
    full = get_rgba(layer)
    if full is None:
        return None
    f = LEVEL_FACTORS[level]
    if f == 1:
        small = full
    else:
        h, w = full.shape[:2]
        im = Image.fromarray(full).resize((max(1, w // f), max(1, h // f)),
                                          Image.BILINEAR)
        small = np.array(im)
    _pyr_cache[key] = small
    return small


def png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ------------------------------------------------------------------ clock
@app.get("/api/clock")
def api_clock():
    return mars_time.clock(JEZERO_LON)


@app.get("/api/meta")
def api_meta():
    # Mars 2020 landing site, planetocentric lat/lon (Octavia E. Butler Landing)
    hab_latlon = [18.4447, 77.4508]

    def lonlat_to_px(lon, lat):
        # inverse of the linear georeference used in scripts/preprocess.py
        px = (lon - BBOX["w"]) / (BBOX["e"] - BBOX["w"]) * W - 0.5
        py = (BBOX["n"] - lat) / (BBOX["n"] - BBOX["s"]) * H - 0.5
        return int(round(px)), int(round(py))

    hab_px = lonlat_to_px(hab_latlon[1], hab_latlon[0])
    layers = [
        dict(id="terrain", name="CTX Terrain", kind="base",
             desc="Hillshaded hypsometric tint of the Mars 2020 CTX 20 m DEM",
             source="USGS/JPL - Mars 2020 Science Investigation CTX DEM"),
        dict(id="ortho", name="CTX Ortho Imagery", kind="base",
             desc="Orthorectified Context Camera mosaic of Jezero",
             source="USGS Astrogeology - CTX ortho mosaic"),
        dict(id="themis", name="THEMIS Day-IR", kind="base",
             desc="Thermal infrared daytime mosaic - rock/sand sensitivity",
             source="NASA Mars Odyssey via Mars Trek WMTS"),
        dict(id="mola", name="MOLA Color Hillshade", kind="base",
             desc="Global laser altimetry colour hillshade for regional context",
             source="NASA Mars Global Surveyor via Mars Trek WMTS"),
        dict(id="hillshade", name="Hillshade", kind="base",
             desc="Raw NW-illumination hillshade (45 deg sun)",
             source="Derived from CTX DEM"),
        dict(id="slopeh", name="Slope Hazard", kind="overlay",
             desc="Slope in degrees - red = >25 deg EVA no-go",
             source="Derived from CTX DEM"),
        dict(id="roughh", name="Roughness", kind="overlay",
             desc="200 m-window elevation std - boulder-field proxy",
             source="Derived from CTX DEM"),
        dict(id="mineral", name="Mineral Prospectivity", kind="overlay",
             desc="CRISM-informed units: clay, carbonate, olivine, pyroxene, silica, sulfate, dune",
             source="Modelled from Jezero geology (CRISM/Keck literature)"),
        dict(id="trav", name="ML Traversability", kind="overlay",
             desc="P(crew-rated traverse) from gradient-boosted terrain model",
             source=f"ML classifier - {metrics.get('traversability_clean_acc', 0) * 100:.1f}% agreement with EVA rule"),
        dict(id="science", name="ML Science Value", kind="overlay",
             desc="Predicted science utility of each cell",
             source=f"ML regressor - val R2 {metrics['science_r2']:.3f}"),
        dict(id="objective", name="ML Objective", kind="overlay",
             desc="science value x traversability - where to walk",
             source="Combined ML objective"),
    ]
    return dict(
        name="Jezero Crater EVA Theatre",
        grid=dict(W=W, H=H, px_m=meta["px_m"], bbox=BBOX),
        elev_range=[round(meta["elev_min"], 1), round(meta["elev_max"], 1)],
        mineral_classes=meta["mineral_classes"],
        sources=meta["sources"],
        layers=layers,
        metrics=metrics,
        verification=verification,
        hab=dict(px=dict(x=hab_px[0], y=hab_px[1]),
                 lat=hab_latlon[0], lon=hab_latlon[1],
                 name="HAB-1 - Octavia E. Butler Landing"),
        targets=targets_doc["targets"],
        levels=LEVEL_FACTORS,
        tile=TILE,
        now_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        mars=mars_time.clock(JEZERO_LON),
    )


@app.get("/api/tiles/{layer}/{level}/{tx}/{ty}.png")
def tile(layer: str, level: int, tx: int, ty: int):
    if layer not in RGBA_FILES and layer not in SCALAR_CMAPS:
        raise HTTPException(404, "unknown layer")
    if not (0 <= level < len(LEVEL_FACTORS)):
        raise HTTPException(404, "bad level")
    small = get_level_rgba(layer, level)
    if small is None:
        raise HTTPException(404, "layer not built")
    sh, sw = small.shape[:2]
    cols = math.ceil(sw / TILE)
    rows = math.ceil(sh / TILE)
    if not (0 <= tx < cols and 0 <= ty < rows):
        return Response(png_bytes(np.zeros((TILE, TILE, 4), np.uint8)),
                        media_type="image/png")
    x0, y0 = tx * TILE, ty * TILE
    x1, y1 = min(x0 + TILE, sw), min(y0 + TILE, sh)
    block = small[y0:y1, x0:x1]
    canvas = np.zeros((TILE, TILE, 4), np.uint8)
    canvas[: block.shape[0], : block.shape[1]] = block
    return Response(png_bytes(canvas), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/probe")
def probe(px: int = Query(...), py: int = Query(...)):
    if not (0 <= px < W and 0 <= py < H):
        raise HTTPException(404, "outside map")
    lon = BBOX["w"] + (px + 0.5) / W * (BBOX["e"] - BBOX["w"])
    lat = BBOX["n"] - (py + 0.5) / H * (BBOX["n"] - BBOX["s"])
    cls = meta["mineral_classes"][int(_mid[py, px])]
    return dict(
        px=px, py=py, lon=round(lon, 5), lat=round(lat, 5),
        elev=round(float(_elev[py, px]), 1),
        slope=round(float(_slope[py, px]), 1),
        rough=round(float(_rough[py, px]), 2),
        aspect=round(float(_aspect[py, px]), 0),
        mineral=cls,
        mineral_conf=round(float(_mconf[py, px]), 3),
        traversability=round(float(_trav[py, px]), 3),
        science=round(float(_sci[py, px]), 2),
    )


# ---------------------------------------------------------------- routes
_planner = None


def get_planner():
    global _planner
    if _planner is None:
        ps = np.load(os.path.join(LAYERS_DIR, "plan_slope.npy"))
        pr = np.load(os.path.join(LAYERS_DIR, "plan_rough.npy"))
        pe = np.load(os.path.join(LAYERS_DIR, "plan_elev.npy"))
        _planner = RoutePlanner(ps, pr, pe, meta)
    return _planner


@app.post("/api/route")
def api_route(req: dict):
    wps = req.get("waypoints") or []
    if len(wps) < 2:
        raise HTTPException(400, "need at least 2 waypoints")
    max_slope = float(req.get("max_slope", 20.0))
    n_crew = int(req.get("crew", 2))
    use_science = bool(req.get("prefer_science", True))
    planner = get_planner()
    sf = planner.plan_factor
    plan_wps = []
    for wp in wps:
        y = int(np.clip(round(wp["y"] / sf), 0, planner.H - 1))
        x = int(np.clip(round(wp["x"] / sf), 0, planner.W - 1))
        plan_wps.append((y, x))
    science_p = (_obj[::sf, ::sf][:planner.H, :planner.W] if use_science else None)

    path, err, notes = planner.plan_multi(plan_wps, max_slope=max_slope,
                                          science=science_p)
    if path is None:
        raise HTTPException(422, err or "infeasible")
    path_work = (np.array(path, np.float64) * sf + sf / 2.0).astype(np.int64)
    path_work[:, 0] = np.clip(path_work[:, 0], 0, H - 1)
    path_work[:, 1] = np.clip(path_work[:, 1], 0, W - 1)
    n_stops = max(0, len(wps) - 2)
    stats = stats_for_path(path_work, _elev, _slope, _rough, _trav, _sci,
                           meta["px_m"], n_crew=n_crew, n_stops=n_stops)

    notes = list(notes)
    # The slope limit is enforced on 80 m planning cells, but the grade below is
    # measured between consecutive 40 m samples along the track, so a short
    # steep step inside an otherwise gentle cell can still show up. Say so
    # rather than letting the two numbers appear to contradict each other.
    if stats and stats["max_grade"] > max_slope:
        notes.append(
            f"steepest 40 m segment reaches {stats['max_grade']:.1f}° against a "
            f"{max_slope:.0f}° limit enforced on {int(planner.px_m)} m planning cells "
            "- inspect the profile before committing")
    stats["warnings"] = list(stats["warnings"]) + notes

    # sample path for client rendering (every ~4th node)
    step = max(1, len(path_work) // 800)
    sample = path_work[::step].tolist()
    if sample[-1] != path_work[-1].tolist():
        sample.append(path_work[-1].tolist())

    # science targets encountered (within 1 km of the planned line)
    tol_px = int(round(1000.0 / meta["px_m"]))
    encountered = []
    for t in targets_doc["targets"]:
        d = np.hypot(path_work[:, 0] - t["py"], path_work[:, 1] - t["px"])
        if d.min() <= tol_px:
            encountered.append(t["id"])

    # cumulative distance at each waypoint (for profile stop markers)
    dr = np.diff(path_work[:, 0].astype(np.float64))
    dc = np.diff(path_work[:, 1].astype(np.float64))
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(dr, dc))]) * meta["px_m"] / 1000.0
    stops_km = []
    for wp in wps:
        d2 = (path_work[:, 0] - wp["y"]) ** 2 + (path_work[:, 1] - wp["x"]) ** 2
        stops_km.append(round(float(cum[int(np.argmin(d2))]), 3))

    return dict(
        path=[dict(y=int(y), x=int(x)) for y, x in sample],
        stats=stats,
        encountered=encountered,
        stops_km=stops_km,
        nodes_explored=int(planner.last_pops),
        plan_grid=dict(W=planner.W, H=planner.H, cell_m=planner.px_m),
        notes=notes,
    )


@app.get("/api/elev/raw")
def elev_raw():
    """Raw float32 elevation grid for the 3D mesh builder."""
    return Response(_elev.astype("<f4").tobytes(),
                    media_type="application/octet-stream",
                    headers={"Content-Length": str(_elev.size * 4)})


# ------------------------------------------------- Perseverance ground truth
_traverse_cache = None
TRAVERSE_PATH = os.path.join(ROOT, "data", "perseverance_traverse.json")


@app.get("/api/rover/traverse")
def api_rover_traverse():
    """The rover's ACTUAL driven track, from NASA/NAIF M2020 SPICE kernels.

    This is our ground truth: real positions published by the mission, reduced
    from the surface-rover-location SPK. It is what the planner will eventually
    be scored against, so the map shows it next to anything the planner
    proposes.
    """
    global _traverse_cache
    if _traverse_cache is None:
        if not os.path.exists(TRAVERSE_PATH):
            raise HTTPException(404, "no traverse; run scripts/fetch_traverse.py")
        with open(TRAVERSE_PATH) as f:
            doc = json.load(f)
        pts = []
        for r in doc.get("track", []):
            px = (r["lon"] - BBOX["w"]) / (BBOX["e"] - BBOX["w"]) * W
            py = (BBOX["n"] - r["lat"]) / (BBOX["n"] - BBOX["s"]) * H
            if -50 <= px <= W + 50 and -50 <= py <= H + 50:
                pts.append(dict(x=round(px, 1), y=round(py, 1), sol=r["sol"]))
        _traverse_cache = dict(
            source=doc.get("source", ""),
            convention=doc.get("convention", ""),
            coverage_utc=doc.get("coverage_utc", []),
            sol_range=doc.get("sol_range", []),
            n_samples=doc.get("n_samples", 0),
            distance_km=doc.get("distance_km", 0),
            grid=dict(W=W, H=H, px_m=meta["px_m"], bbox=BBOX),
            points=pts,
        )
    return _traverse_cache


@app.get("/api/targets")
def api_targets():
    return targets_doc


@app.get("/api/live/photos")
def api_photos():
    return nasa_live.get_photos()


@app.get("/api/live/image")
def api_live_image(u: str = Query(...), id: str = Query("")):
    """Serve a NASA image from our own origin, cached on disk.

    The gallery previously hot-linked images-assets.nasa.gov directly, which
    works in a plain browser but is blocked by client-side network policies and
    is at the mercy of venue Wi-Fi. Proxying through the server also means the
    bytes are cached, so the second view is instant and a later offline demo
    still shows imagery.
    """
    try:
        data, ctype = nasa_live.fetch_photo_bytes(u, id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"upstream image fetch failed: {e}")
    return Response(data, media_type=ctype,
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/live/conditions")
def api_conditions():
    return nasa_live.conditions(mars_time.clock(JEZERO_LON))


@app.get("/api/live/weather")
def api_weather():
    """Raw measured surface weather, kept separate so it can be checked."""
    return nasa_live.mars_weather()


@app.get("/api/health")
def health():
    return dict(ok=True, grid=[W, H], layers=len(RGBA_FILES) + len(SCALAR_CMAPS),
                targets=len(targets_doc["targets"]),
                sol=mars_time.clock(JEZERO_LON)["sol"])


# ---------------------------------------------------------------- static
_fallback_dir = os.path.join(ROOT, "data", "fallback_photos")
os.makedirs(_fallback_dir, exist_ok=True)
app.mount("/fl", StaticFiles(directory=_fallback_dir), name="fallback")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    with open(os.path.join(STATIC, "index.html")) as f:
        return Response(f.read(), media_type="text/html")
