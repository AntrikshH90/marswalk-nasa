#!/usr/bin/env python3
"""Extract the REAL Perseverance traverse from NASA/NAIF SPICE kernels.

This is the project's ground truth. The rover's actual position over the whole
surface mission is published by NAIF in the M2020 "surface rover location"
SPK kernels, so instead of guessing at traversability from an invented rule we
can ask what terrain the rover actually drove through - and later measure
whether our planner reproduces those drives.

What it does:
  1. downloads the M2020 SPICE kernels needed to reduce position (SPK + LSK +
     PCK), with retries, into data/spice/
  2. samples the rover body (-168) in the Mars body-fixed frame (IAU_MARS)
  3. writes the driven track to data/perseverance_traverse.json

Coordinates come out planetocentric, positive east, which is the same
convention the DEM footprint uses, so the track lands in the app's pixel frame
without any reprojection.

Run:  python3 scripts/fetch_traverse.py
      python3 scripts/fetch_traverse.py --step-min 20      # coarser sampling
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SPICE_DIR = os.path.join(ROOT, "data", "spice")
OUT = os.path.join(ROOT, "data", "perseverance_traverse.json")

BASE = "https://naif.jpl.nasa.gov/pub/naif/MARS2020/kernels/"
UA = {"User-Agent": "marswalk-hackathon/1.0 (NASA Space Apps Challenge)"}

ROVER = -168            # M2020 / Perseverance
MARS = 499
MARS_FRAME = "IAU_MARS"  # body-fixed, planetocentric, positive east

NEEDED = ([("spk", "m2020_surf_rover_loc.bsp")]
          + [("spk", f"m2020_surf_rover_loc_{a:04d}_{b:04d}_v1.bsp")
             for a, b in [(0, 89), (89, 179), (179, 299), (299, 419), (419, 539),
                          (539, 659), (659, 779), (779, 899), (899, 1019),
                          (1019, 1139), (1139, 1259), (1259, 1379), (1379, 1499),
                          (1499, 1619), (1619, 1739), (1739, 1859)]]
          + [("lsk", "naif0012.tls"), ("pck", "pck00010.tpc")])


def log(*a):
    print(*a, flush=True)


def download_kernels(force=False):
    os.makedirs(SPICE_DIR, exist_ok=True)
    got = 0
    for sub, name in NEEDED:
        dest = os.path.join(SPICE_DIR, name)
        if not force and os.path.exists(dest) and os.path.getsize(dest) > 512:
            continue
        url = f"{BASE}{sub}/{name}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers=UA), timeout=180) as r:
                    blob = r.read()
                with open(dest, "wb") as f:
                    f.write(blob)
                got += 1
                log(f"    {len(blob) / 1e6:7.2f} MB  {name}")
                break
            except Exception as e:
                if attempt == 4:
                    log(f"    FAILED {name}: {type(e).__name__} {e}")
                else:
                    time.sleep(3 + 4 * attempt)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-min", type=float, default=10.0,
                    help="sampling cadence in minutes (default 10)")
    ap.add_argument("--move-m", type=float, default=0.30,
                    help="keep a sample only if the rover moved this far (m)")
    ap.add_argument("--force", action="store_true", help="re-download kernels")
    a = ap.parse_args()

    log("[1/3] SPICE kernels …")
    n = download_kernels(a.force)
    log(f"    {n} newly downloaded; {len(os.listdir(SPICE_DIR))} files present")

    try:
        import spiceypy as sp
    except ImportError:
        log("spiceypy missing.  pip install spiceypy")
        return 1

    sp.kclear()
    for sub, name in NEEDED:
        p = os.path.join(SPICE_DIR, name)
        if os.path.exists(p) and name.endswith((".bsp", ".tls", ".tpc")):
            try:
                sp.furnsh(p)
            except Exception as e:
                log(f"    warn: could not load {name}: {e}")

    log("[2/3] sampling rover position …")
    cell = sp.cell_double(64)
    merged = os.path.join(SPICE_DIR, "m2020_surf_rover_loc.bsp")
    sp.spkcov(merged, ROVER, cell)
    m = sp.card(cell)
    if not m:
        log("    no coverage for the rover body in the merged kernel")
        return 1
    et0, et1 = cell[0], cell[1]

    # Mars radius reference from the PCK, used only to report an elevation
    try:
        radii = sp.bodvrd("MARS", "RADII", 3)[1]
        r_km = float(radii[0])
    except Exception:
        r_km = 3396.19

    step = a.step_min * 60.0
    pts = []
    t = et0
    last_kept = None
    while t <= et1:
        try:
            pos, _ = sp.spkpos(str(ROVER), t, MARS_FRAME, "NONE", str(MARS))
        except Exception:
            t += step
            continue
        x, y, z = pos
        r = math.sqrt(x * x + y * y + z * z)
        lon = math.degrees(math.atan2(y, x))
        lat = math.degrees(math.asin(z / r))
        if last_kept is None:
            keep = True
        else:
            d = 1000.0 * math.sqrt((x - last_kept[0]) ** 2 + (y - last_kept[1]) ** 2
                                   + (z - last_kept[2]) ** 2)
            keep = d >= a.move_m
        if keep:
            pts.append((t, lon, lat, (r - r_km) * 1000.0, x, y, z))
            last_kept = (x, y, z)
        t += step

    log(f"    {len(pts)} moved samples over the mission")

    log("[3/3] writing traverse …")
    from app import mars_time as mt

    recs = []
    total_m = 0.0
    prev = None
    for (t, lon, lat, elev, x, y, z) in pts:
        utc = sp.et2utc(t, "ISOC", 0)
        dt = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        if prev is not None:
            total_m += 1000.0 * math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2
                                          + (z - prev[2]) ** 2)
        prev = (x, y, z)
        recs.append(dict(
            et=round(t, 1), utc=utc.replace("+00:00", "Z"),
            sol=mt.sol_number(dt), ls=round(mt.ls_deg(dt), 2),
            lon=round(lon, 6), lat=round(lat, 6), elev_m=round(elev, 1),
        ))

    doc = dict(
        source=("NASA/NAIF M2020 surface rover location SPK kernels, body -168, "
                "frame IAU_MARS, target Mars (499)"),
        kernels=[n for _, n in NEEDED],
        convention="planetocentric latitude, positive east longitude",
        coverage_utc=[recs[0]["utc"], recs[-1]["utc"]] if recs else [],
        n_samples=len(recs),
        sol_range=[recs[0]["sol"], recs[-1]["sol"]] if recs else [],
        distance_km=round(total_m / 1000.0, 3),
        bbox_lonlat=([min(r["lon"] for r in recs), min(r["lat"] for r in recs),
                      max(r["lon"] for r in recs), max(r["lat"] for r in recs)]
                     if recs else []),
        note=("elev_m is height above the MARS PCK reference sphere, not above the "
              "DEM geoid, so it differs from the DEM by a constant offset; the "
              "horizontal track is what matters here"),
        track=recs,
    )
    with open(OUT, "w") as f:
        json.dump(doc, f)
    log(f"    {OUT}")
    log(f"    sols {doc['sol_range']}  samples {doc['n_samples']}  "
        f"driven {doc['distance_km']} km")
    log(f"    bbox lon/lat {[round(v, 4) for v in doc['bbox_lonlat']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
