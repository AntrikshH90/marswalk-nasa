#!/usr/bin/env python3
"""
MARSWALK — Preprocessing pipeline
Builds the layered raster stack for Jezero Crater from real mission data:

  * CTX 20 m DEM (Mars 2020 Science Investigation mosaic, USGS/JPL)  -> elevation, slope,
    roughness, curvature, hillshade  [REAL]
  * THEMIS Day-IR 100 m mosaic (Mars Odyssey) via NASA Mars Trek WMTS  -> context imagery [REAL]
  * MGS MOLA color hillshade (Mars Global Surveyor) via Mars Trek     -> global context [REAL]
  * CTX 5 m ortho browse mosaic                                       -> photographic base [REAL]
  * CRISM-informed mineral prospectivity model                       -> modelled layer
Outputs float32/RGBA arrays in layers/ + georeferencing metadata.
"""
import json
import math
import os
import sys
import urllib.request

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LAYERS = os.path.join(ROOT, "layers")
os.makedirs(LAYERS, exist_ok=True)

# ---------------------------------------------------------------- georef
# M20_JezeroCrater_CTXDEM_20m.tif is delivered in "Jezero_Crater_Projection"
# (a local equirectangular projection on the Mars 2000 sphere), NOT in lon/lat,
# so the pixel grid is georeferenced by its published corner coordinates.
#
# Provenance of the numbers below: the product's own metadata (footprint
# west/east/south/north) and its GeoTIFF tags (ModelPixelScale 20.0 m,
# 4456 x 5067 px). The projection scales longitude by cos(centre latitude)
# = cos(18.435 deg) = 0.94868, which makes pixel <-> lon/lat exactly linear:
#
#   x: 4456 px * 20 m = 89,120 m  vs  1.585126 deg * 59274.6 m/deg * 0.94868
#                                         = 89,133 m   (0.015 % agreement)
#   y: 5067 px * 20 m = 101,340 m vs  1.709667 deg * 59274.6 m/deg
#                                         = 101,339 m  (0.001 % agreement)
#
# so the linear mapping used throughout is accurate to a few metres, and the
# check is re-printed on every run (see verify_georef).
BBOX = dict(w=76.997437, e=78.582563, n=19.290187, s=17.580520)
W0, H0 = 4456, 5067          # native DEM size @ 20 m/px
FACTOR = 2                    # working grid factor
W, H = W0 // FACTOR, H0 // FACTOR + 1   # 2228 x 2534 working grid (40 m/px)
PX_M = 20.0 * FACTOR

# Mars 2000 sphere radius used by the product's projection
MARS_RADIUS_M = 3396190.0
DEG_M = 2 * 3.14159265358979 * MARS_RADIUS_M / 360.0   # metres per degree

UA = {"User-Agent": "marswalk-hackathon/1.0 (NASA Space Apps Challenge)"}


def log(*a):
    print(*a, flush=True)


def verify_georef():
    """Re-check the linear pixel -> lon/lat mapping against the raster size."""
    lon_m = (BBOX["e"] - BBOX["w"]) * DEG_M * math.cos(math.radians((BBOX["n"] + BBOX["s"]) / 2))
    lat_m = (BBOX["n"] - BBOX["s"]) * DEG_M
    x_m, y_m = W0 * 20.0, H0 * 20.0
    log(f"  georef check: x {x_m:,.0f} m raster vs {lon_m:,.0f} m footprint "
        f"({abs(x_m - lon_m) / lon_m * 100:.3f}%)")
    log(f"                y {y_m:,.0f} m raster vs {lat_m:,.0f} m footprint "
        f"({abs(y_m - lat_m) / lat_m * 100:.3f}%)")



def px_to_lonlat(px, py):
    lon = BBOX["w"] + (px + 0.5) / W * (BBOX["e"] - BBOX["w"])
    lat = BBOX["n"] - (py + 0.5) / H * (BBOX["n"] - BBOX["s"])
    return float(lon), float(lat)


# ---------------------------------------------------------------- 1. DEM
def load_dem():
    """Read the delivery DEM, fill void pixels, and downsample to the working grid.

    The mosaic carries ~14k void pixels (0.06 %), essentially all of them in a
    border frame where the CTX stereo coverage stops. Pixels are filled from
    the *nearest valid* neighbour via a distance transform.

    (The previous implementation filled from a 5x5 window mean and substituted
    0.0 wherever that window was entirely void. On Mars, 0 m is about 2,300 m
    above the terrain here, so any wide void region would have been seeded
    with a spike roughly the height of a small mountain. Propagating real
    values is both simpler and correct.)
    """
    path = os.path.join(DATA, "jezero_dem_20m.tif")
    try:
        arr = tifffile.imread(path).astype(np.float32)
    except Exception:
        from PIL import Image as _I
        _I.MAX_IMAGE_PIXELS = None
        arr = np.array(_I.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]

    nodata = ~np.isfinite(arr) | (arr < -1e30)
    n_void = int(nodata.sum())
    if n_void:
        idx = ndimage.distance_transform_edt(nodata, return_distances=False,
                                             return_indices=True)
        arr = arr[tuple(idx)]
        log(f"  filled {n_void} void px ({100.0 * n_void / arr.size:.3f}%) from nearest valid neighbour")
    if n_void and n_void == arr.size:
        raise RuntimeError("DEM is entirely void - refusing to guess elevation")

    # downsample x2 by block mean (2228 x 2533) plus a one-row edge pad, since
    # 5067 is odd and the working grid is defined as 2534 rows tall
    h_even = (H0 // FACTOR) * FACTOR
    blocks = arr[:h_even].reshape(h_even // FACTOR, FACTOR, W0 // FACTOR, FACTOR)
    small = blocks.mean(axis=(1, 3)).astype(np.float32)
    if H > small.shape[0]:
        small = np.vstack([small, small[-1:]])
    return small[:H, :W]


def derive_terrain(elev):
    gy, gx = np.gradient(elev, PX_M)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    aspect = np.degrees(np.arctan2(gx, -gy)).astype(np.float32)  # 0=N
    # roughness: local std in 5x5 window (200 m)
    mean = ndimage.uniform_filter(elev, 5)
    mean2 = ndimage.uniform_filter(elev * elev, 5)
    rough = np.sqrt(np.maximum(mean2 - mean * mean, 0)).astype(np.float32)
    curv = ndimage.laplace(elev) / (PX_M * PX_M) * 1000.0  # m/km^2 scaled
    curv = curv.astype(np.float32)
    # hillshade (sun from NW 315deg, alt 45deg)
    az, alt = np.radians(315.0), np.radians(45.0)
    gx_n = gx / np.maximum(np.hypot(gx, gy), 1e-6)
    gy_n = gy / np.maximum(np.hypot(gx, gy), 1e-6)
    hs = (np.cos(alt) * np.sin(az) * -gx_n
          + np.cos(alt) * np.cos(az) * gy_n * -1
          + np.sin(alt) * np.sqrt(np.maximum(1 - 0.0, 0)))
    # proper normal-based shading
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny = -gx * nz, -gy * nz
    shade = np.clip(nx * np.cos(alt) * np.sin(az) + ny * np.cos(alt) * np.cos(az)
                    + nz * np.sin(alt), 0, 1)
    hillshade = (shade * 255).astype(np.uint8)
    return slope, aspect, rough, curv, hillshade


# ------------------------------------------------------- 2. Trek mosaics
TREK = {
    "themis": dict(
        layer="THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018",
        ext="png", best_z=[9, 8, 7, 6]),
    "mola": dict(
        layer="Mars_MGS_MOLA_ClrShade_merge_global_463m",
        ext="jpg", best_z=[8, 7, 6]),
}


def trek_tile_url(spec, z, row, col):
    return (f"https://trek.nasa.gov/tiles/Mars/EQ/{spec['layer']}"
            f"/1.0.0/default/default028mm/{z}/{row}/{col}.{spec['ext']}")


def fetch(url, dest, retries=2):
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        return True
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            if len(data) > 500 and not data[:60].lower().startswith(b"<!"):
                with open(dest, "wb") as f:
                    f.write(data)
                return True
        except Exception as e:
            if i == retries:
                log(f"  fetch fail {url}: {e}")
    return False


def mosaic_trek(name):
    """Compose the trek layer over our bbox into a full-res image, then resize to grid.

    If the layer was built on a previous run we reuse it: the source mosaics are
    static for this project, and re-fetching hundreds of tiles over a slow link
    only risks producing a worse result than the one already on disk. Pass
    --force-mosaics to rebuild from the network.
    """
    spec = TREK[name]
    out_path = os.path.join(LAYERS, f"{name}.png")
    if not FORCE_MOSAICS and os.path.exists(out_path):
        try:
            im = Image.open(out_path).convert("RGB")
            log(f"  {name}: reusing existing {name}.png {im.size}")
            return im if im.size == (W, H) else im.resize((W, H), Image.BILINEAR)
        except Exception as e:
            log(f"  {name}: existing file unusable ({e}), refetching")
    cache = os.path.join(DATA, f"trek_{name}")
    os.makedirs(cache, exist_ok=True)
    for z in spec["best_z"]:
        rows, cols = 2 ** z, 2 ** (z + 1)
        def rc(lon, lat):
            c = int((lon + 180) / 360 * cols)
            r = int((90 - lat) / 180 * rows)
            return min(max(r, 0), rows - 1), min(max(c, 0), cols - 1)
        r0, c0 = rc(BBOX["w"], BBOX["n"])          # top-left
        r1, c1 = rc(BBOX["e"], BBOX["s"])          # bottom-right
        # ensure ascending
        r0, r1 = min(r0, r1), max(r0, r1)
        c0, c1 = min(c0, c1), max(c0, c1)
        ok = True
        files = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                dest = os.path.join(cache, f"z{z}_{r}_{c}.{spec['ext']}")
                if not fetch(trek_tile_url(spec, z, r, c), dest):
                    ok = False
                    break
                files[(r, c)] = dest
            if not ok:
                break
        if not ok or not files:
            log(f"  {name}: z{z} failed, trying coarser")
            continue
        # compose
        tw = th = 256
        nr, nc = r1 - r0 + 1, c1 - c0 + 1
        canvas = Image.new("RGB", (nc * tw, nr * th))
        for (r, c), f in files.items():
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                im = Image.new("RGB", (tw, th), (0, 0, 0))
            canvas.paste(im, ((c - c0) * tw, (r - r0) * th))
        # georeference of mosaic in lon/lat
        deg = 360.0 / (2 * (2 ** z))  # tile angular size = 360/cols
        west = -180.0 + c0 * (360.0 / nc_total(nc, c0)) if False else None
        # simpler: columns span from tile col c0 left edge
        left = -180.0 + c0 * (360.0 / (2 * (2 ** z)))
        top = 90.0 - r0 * (180.0 / (2 ** z))
        right = -180.0 + (c1 + 1) * (360.0 / (2 * (2 ** z)))
        bottom = 90.0 - (r1 + 1) * (180.0 / (2 ** z))
        # crop to BBOX then resize to grid
        span_x = (right - left)
        span_y = (top - bottom)
        box = (int(round((BBOX['w'] - left) / span_x * canvas.width)),
               int(round((top - BBOX['n']) / span_y * canvas.height)),
               int(round((BBOX['e'] - left) / span_x * canvas.width)),
               int(round((top - BBOX['s']) / span_y * canvas.height)))
        box = (max(0, box[0]), max(0, box[1]),
               min(canvas.width, box[2]), min(canvas.height, box[3]))
        if box[2] - box[0] < 10 or box[3] - box[1] < 10:
            continue
        cropped = canvas.crop(box).resize((W, H), Image.BILINEAR)
        log(f"  {name}: mosaicked from z{z} ({len(files)} tiles)")
        return cropped
    log(f"  {name}: all zooms failed")
    return None


def nc_total(*a):  # placeholder (kept for readability above)
    return 1


def load_ortho():
    path = os.path.join(DATA, "jezero_ortho_browse.jpg")
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    return im.resize((W, H), Image.BILINEAR)


# --------------------------------------------------- 3. mineral model
def fractal_noise(shape, octaves=6, seed=7):
    """Fast 1/f noise via upsampled random-octave pyramid (PIL bilinear)."""
    rng = np.random.default_rng(seed)
    H, W = shape
    total = np.zeros(shape, np.float32)
    amp, norm = 1.0, 0.0
    base = 4
    for o in range(octaves):
        gh = min(H, int(base * (2 ** o)))
        gw = min(W, int(base * (2 ** o) * W / H))
        gh, gw = max(gh, 4), max(gw, 4)
        small = rng.standard_normal((gh, gw)).astype(np.float32)
        small = ndimage.gaussian_filter(small, 0.9, mode="wrap")
        im = Image.fromarray(small, mode="F").resize((W, H), Image.BILINEAR)
        total += np.asarray(im, np.float32) * amp
        norm += amp
        amp *= 0.55
    total /= (norm or 1.0)
    total /= (total.std() or 1.0)
    return total.astype(np.float32)


def mineral_model(elev, slope, rough):
    """CRISM-informed mineral prospectivity for Jezero.

    Physically placed units (peer-reviewed Jezero geology):
      - Western fan & delta: Mg-Fe phyllosilicates + carbonate
      - Northern/western inflow channels: altered clays
      - Crater floor: mafic pyroxene-rich basaltic sand/bedrock
      - Margin units (SE): olivine + carbonate (Nili/Faxa unit analogues)
      - Southern high ground: silica/phyllosilicate bedrock prospects
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lonlat = np.array([[px_to_lonlat(x, y) for x in (0, W - 1)] for y in (0, H - 1)])
    # normalized coords
    u = xx / (W - 1)          # 0..1 west->east
    v = yy / (H - 1)          # 0..1 north->south
    lon = BBOX["w"] + u * (BBOX["e"] - BBOX["w"])
    lat = BBOX["n"] - v * (BBOX["n"] - BBOX["s"])

    n1 = fractal_noise((H, W), 6, seed=11)
    n2 = fractal_noise((H, W), 6, seed=23)
    n3 = fractal_noise((H, W), 5, seed=41)

    elev_n = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
    low = 1.0 - np.clip((elev_n - 0.25) / 0.5, 0, 1)   # paleolake lowlands

    # geometry approximations (normalized): W inlet channel ~ mid-north-west fan lobe
    # western fan: centered u~0.22, v~0.42 (west of basin), lobe radius
    fan = np.exp(-(((u - 0.24) / 0.16) ** 2 + ((v - 0.44) / 0.20) ** 2) * 2.2)
    # northern channel
    chan_n = np.exp(-((u - 0.40) / 0.05) ** 2) * np.clip((0.45 - v) / 0.45, 0, 1)
    # western channel
    chan_w = np.exp(-((v - 0.30) / 0.05) ** 2) * np.clip((0.35 - u) / 0.35, 0, 1)
    # southern outflow / raised rim
    south = np.clip((v - 0.78) / 0.22, 0, 1)
    # southeastern margin unit (olivine-carbonate)
    margin = np.exp(-(((u - 0.74) / 0.14) ** 2 + ((v - 0.66) / 0.14) ** 2) * 2.0)
    # crater rim ring
    cu, cv = (u - 0.5) * 1.75, (v - 0.52)
    ring = np.exp(-((np.hypot(cu, cv) - 0.62) / 0.09) ** 2)

    scores = {}
    scores["phyllosilicate"] = 1.6 * fan + 0.9 * chan_n + 0.7 * chan_w + 0.5 * low + 0.35 * n1
    scores["carbonate"] = 1.3 * fan + 1.1 * margin + 0.4 * ring + 0.3 * n2 + 0.25 * low
    scores["olivine"] = 1.5 * margin + 0.6 * south + 0.35 * ring + 0.3 * n2
    scores["pyroxene"] = 1.0 + 0.7 * (1 - low) + 0.3 * n3 - 0.4 * fan
    scores["silica"] = 1.2 * south + 0.6 * chan_n + 0.4 * n3 + 0.2 * (elev_n > 0.75)
    scores["sulfate"] = 0.9 * low + 0.5 * ring + 0.3 * n1 - 0.2 * fan
    # dune sand (physical, from low-lying rough texture)
    scores["dune_sand"] = 1.4 * low * np.clip((elev_n - 0.1) / 0.4, 0, 1) + 0.3 * n2 - 0.5 * slope / 30.0

    names = list(scores)
    # memory-safe softmax over channels (2D temps only)
    maxv = None
    argmax = np.zeros((H, W), np.uint8)
    for i, k in enumerate(names):
        s = np.clip(scores[k], 0, None)
        if maxv is None:
            maxv = s.copy()
            argmax[:] = i
        else:
            m = s > maxv
            argmax[m] = i
            maxv[m] = s[m]
        scores[k] = s  # keep clipped, free original object if distinct
    denom = np.zeros((H, W), np.float32)
    for k in names:
        denom += np.exp(scores[k] - maxv)
    # winning class has score == maxv => exp(0)=1 => prob = 1 / denom
    maxp = (1.0 / np.maximum(denom, 1e-6)).astype(np.float32)
    return names, argmax, maxp, None


PALETTE = {
    "phyllosilicate": (126, 200, 140),
    "carbonate": (240, 214, 120),
    "olivine": (176, 208, 74),
    "pyroxene": (132, 106, 178),
    "silica": (99, 205, 214),
    "sulfate": (244, 156, 99),
    "dune_sand": (224, 178, 120),
}


def save_rgba(path, rgba):
    Image.fromarray(rgba, "RGBA").save(path)


# set from the command line in main()
FORCE_MOSAICS = False


def main():
    global FORCE_MOSAICS
    FORCE_MOSAICS = "--force-mosaics" in sys.argv

    log("[0/5] checking georeference …")
    verify_georef()

    log("[1/5] loading DEM …")
    elev = load_dem()
    log(f"  grid {elev.shape} elev {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} m "
        f"(mean {np.nanmean(elev):.0f} m)")
    if not (-6000.0 < np.nanmin(elev) < np.nanmax(elev) < 0.0):
        raise RuntimeError(f"elevation range {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} "
                           "is not plausible for Mars - check the DEM and its void fill")


    log("[2/5] deriving terrain …")
    slope, aspect, rough, curv, hillshade = derive_terrain(elev)
    log(f"  slope mean {slope.mean():.1f}° p99 {np.percentile(slope,99):.1f}°")

    log("[3/5] fetching Mars Trek mosaics …")
    themis = mosaic_trek("themis")
    mola = mosaic_trek("mola")
    ortho = load_ortho()

    log("[4/5] mineral prospectivity …")
    mnames, midx, mconf, mprob = mineral_model(elev, slope, rough)
    log(f"  classes: {mnames}")

    log("[5/5] writing arrays …")
    np.save(os.path.join(LAYERS, "elev.npy"), elev)
    np.save(os.path.join(LAYERS, "slope.npy"), slope)
    np.save(os.path.join(LAYERS, "aspect.npy"), aspect)
    np.save(os.path.join(LAYERS, "rough.npy"), rough)
    np.save(os.path.join(LAYERS, "curv.npy"), curv)
    np.save(os.path.join(LAYERS, "hillshade.npy"), hillshade)
    np.save(os.path.join(LAYERS, "mineral_idx.npy"), midx)
    np.save(os.path.join(LAYERS, "mineral_conf.npy"), mconf)

    # display rasters
    # hypsometric tint (mars palette: deep -> high)
    e = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
    stops = np.array([[0.00, 42, 24, 20],
                      [0.22, 96, 43, 26],
                      [0.45, 156, 78, 39],
                      [0.68, 196, 122, 62],
                      [0.86, 222, 168, 105],
                      [1.00, 245, 220, 176]], np.float32)
    hyp = np.zeros((H, W, 3), np.uint8)
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        m = (e >= a[0]) & (e <= b[0])
        t = np.zeros_like(e)
        t[m] = (e[m] - a[0]) / (b[0] - a[0] + 1e-9)
        for ch in range(3):
            hyp[..., ch][m] = (a[1 + ch] + t[m] * (b[1 + ch] - a[1 + ch])).astype(np.uint8)
    # hillshade multiply
    hs = hillshade[..., None].astype(np.float32) / 255.0
    tinted = (hyp.astype(np.float32) * (0.45 + 0.75 * hs)).clip(0, 255).astype(np.uint8)
    rgba_hyp = np.dstack([tinted, np.full((H, W), 255, np.uint8)])
    save_rgba(os.path.join(LAYERS, "terrain.png"), rgba_hyp)

    # hillshade grayscale RGBA
    hs_rgb = np.dstack([hillshade, hillshade, hillshade, np.full((H, W), 255, np.uint8)])
    save_rgba(os.path.join(LAYERS, "hillshade.png"), hs_rgb)

    # slope hazard RGBA (transparent -> yellow -> red)
    sm = np.clip(slope / 35.0, 0, 1)
    rgba_slope = np.zeros((H, W, 4), np.uint8)
    rgba_slope[..., 0] = np.clip(sm * 2 * 255, 0, 255)
    rgba_slope[..., 1] = np.clip((1 - sm) * 200 + 30, 0, 255) * (sm > 0.04)
    rgba_slope[..., 2] = 30
    rgba_slope[..., 3] = np.where(sm > 0.04, np.clip(40 + sm * 215, 0, 255), 0).astype(np.uint8)
    save_rgba(os.path.join(LAYERS, "slope.png"), rgba_slope)

    # roughness RGBA
    rq = np.clip(rough / np.percentile(rough, 98), 0, 1)
    rgba_rq = np.zeros((H, W, 4), np.uint8)
    rgba_rq[..., 0] = 60
    rgba_rq[..., 1] = np.clip(rq * 230, 0, 255)
    rgba_rq[..., 2] = np.clip(120 + rq * 100, 0, 255)
    rgba_rq[..., 3] = np.where(rq > 0.05, np.clip(30 + rq * 200, 0, 255), 0).astype(np.uint8)
    save_rgba(os.path.join(LAYERS, "rough.png"), rgba_rq)

    # mineral RGBA
    rgba_m = np.zeros((H, W, 4), np.uint8)
    for i, name in enumerate(mnames):
        col = PALETTE[name]
        m = midx == i
        for ch in range(3):
            rgba_m[..., ch][m] = col[ch]
    rgba_m[..., 3] = (110 + mconf * 120).astype(np.uint8)
    save_rgba(os.path.join(LAYERS, "mineral.png"), rgba_m)

    if themis is not None:
        save_rgba(os.path.join(LAYERS, "themis.png"),
                  np.dstack([np.array(themis), np.full((H, W), 255, np.uint8)]))
    if mola is not None:
        save_rgba(os.path.join(LAYERS, "mola.png"),
                  np.dstack([np.array(mola), np.full((H, W), 255, np.uint8)]))
    if ortho is not None:
        save_rgba(os.path.join(LAYERS, "ortho.png"),
                  np.dstack([np.array(ortho), np.full((H, W), 255, np.uint8)]))

    # planning grid (80 m) for fast A*
    def block_mean(a, f=2):
        h = (a.shape[0] // f) * f
        w = (a.shape[1] // f) * f
        if a.ndim == 2:
            return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
        return a[:h, :w].reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))
    np.save(os.path.join(LAYERS, "plan_slope.npy"), block_mean(slope).astype(np.float32))
    np.save(os.path.join(LAYERS, "plan_rough.npy"), block_mean(rough).astype(np.float32))
    np.save(os.path.join(LAYERS, "plan_elev.npy"), block_mean(elev).astype(np.float32))

    meta = dict(
        bbox=BBOX, W=W, H=H, W0=W0, H0=H0, factor=FACTOR, px_m=PX_M,
        plan_factor=FACTOR * 2,
        plan_W=W // 2, plan_H=H // 2,
        elev_min=float(elev.min()), elev_max=float(elev.max()),
        mineral_classes=mnames,
        sources=[
            "CTX 20 m DEM — Mars 2020 Science Investigation Mosaic (USGS/JPL)",
            "THEMIS Day-IR — Mars Odyssey via NASA Mars Trek WMTS",
            "MGS MOLA Color Hillshade via NASA Mars Trek WMTS",
            "CTX 5 m ortho browse mosaic (USGS Astrogeology)",
            "CRISM-informed mineral prospectivity (modelled from Jezero geology)",
        ],
    )
    with open(os.path.join(LAYERS, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("done ->", LAYERS)


if __name__ == "__main__":
    main()
