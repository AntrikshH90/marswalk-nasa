#!/usr/bin/env python3
"""Resumable downloader for the large NASA products.

The link this project was built on runs at roughly 0.5 Mbit/s, so a 1.8 GB
HiRISE DTM is a multi-hour download that will not survive a dropped
connection. This fetches in chunks with HTTP Range, appends to a .part file,
and picks up where it left off if you re-run it. It is safe to Ctrl-C and
re-run, and safe to run in the background.

    python3 scripts/fetch_big.py --list
    python3 scripts/fetch_big.py hirise_dtm
    python3 scripts/fetch_big.py hirise_dtm ctx_ortho_5m
    python3 scripts/fetch_big.py --all

Progress is printed every few seconds, and a .part file that is already the
full size is simply renamed into place.
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
UA = {"User-Agent": "marswalk-hackathon/1.0 (NASA Space Apps Challenge)"}
BASE = "https://planetarymaps.usgs.gov/mosaic/mars2020_trn"

# name -> (path under the USGS service, local filename, expected bytes or None)
PRODUCTS = {
    "hirise_dtm": (
        "HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif",
        "jezero_hirise_dtm_1m.tif", 1839545409,
        "1 m/px HiRISE stereo DTM, 21488 x 21400 px = 21.5 x 21.4 km centred on "
        "the landing site. This is the resolution at which boulders and metre "
        "scarps are actually visible, i.e. the terrain that stops rovers."),
    "ctx_ortho_5m": (
        "CTX/ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXortho_mosaic_5m.tif",
        "jezero_ortho_5m.tif", 359599280,
        "5 m/px CTX orthomosaic. Replaces the low-resolution browse JPEG that "
        "the ortho map layer currently uses."),
    "ctx_dem_20m": (
        "CTX/ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXDEM_20m.tif",
        "jezero_dem_20m.tif", 90345300,
        "20 m/px CTX DEM - already present; kept here so one script can restore "
        "every source product."),
    "trn_dtm_20m": (
        "CTX/JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif",
        "jezero_trn_dtm_20m.tif", 9662387,
        "Pre-reprojected equirectangular CTX DTM, small. Useful as an "
        "independent second DEM to cross-check the main one."),
}


def human(n):
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def remote_size(url):
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA, method="HEAD"), timeout=30)
        return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def fetch(key, quiet=False):
    rel, fname, expect, note = PRODUCTS[key]
    url = f"{BASE}/{rel}"
    dest = os.path.join(DATA, fname)
    part = dest + ".part"

    total = remote_size(url)
    if not total and expect:
        total = expect
    if os.path.exists(dest) and total and os.path.getsize(dest) == total:
        print(f"[{key}] already complete: {dest} ({human(total)})")
        return True

    have = os.path.getsize(part) if os.path.exists(part) else 0
    if total and have > total:          # truncated/odd part file - start over
        have = 0
        os.remove(part)

    print(f"[{key}] {human(total) if total else 'size unknown'} -> {os.path.basename(dest)}")
    if not quiet:
        print(f"        {note}")
    if have:
        print(f"        resuming at {human(have)}")

    t0 = time.time()
    last = t0
    mode = "ab" if have else "wb"
    try:
        req = urllib.request.Request(url, headers=UA)
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=120) as r, open(part, mode) as f:
            done = have
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last >= 10:
                    last = now
                    rate = (done - have) / max(now - t0, 1e-6)
                    pct = f"{100.0 * done / total:5.1f}%" if total else "  ?  "
                    eta = ((total - done) / rate / 60.0) if (total and rate > 0) else 0
                    print(f"        {pct} {human(done)}  {rate / 1024:.0f} KB/s  "
                          f"eta {eta:.0f} min", flush=True)
    except KeyboardInterrupt:
        print(f"\n        interrupted - re-run the same command to resume from "
              f"{human(os.path.getsize(part))}")
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"        connection dropped ({type(e).__name__}); "
              f"re-run the same command to resume from "
              f"{human(os.path.getsize(part))}")
        return False

    got = os.path.getsize(part)
    if total and got != total:
        print(f"        incomplete: {human(got)} of {human(total)} - re-run to resume")
        return False
    os.replace(part, dest)
    print(f"        done: {dest} ({human(got)}) in {(time.time() - t0) / 60:.1f} min")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("products", nargs="*", help="product keys, or --all")
    ap.add_argument("--all", action="store_true", help="fetch every product")
    ap.add_argument("--list", action="store_true", help="show products and exit")
    a = ap.parse_args()

    if a.list or (not a.products and not a.all):
        print("available products:\n")
        for k, (rel, fname, expect, note) in PRODUCTS.items():
            mark = " (present)" if os.path.exists(os.path.join(DATA, fname)) else ""
            print(f"  {k:15s} {human(expect) if expect else '?':>9}  {fname}{mark}")
            print(f"  {'':15s} {note}\n")
        print(f"downloads land in {DATA}")
        return 0

    keys = list(PRODUCTS) if a.all else a.products
    bad = [k for k in keys if k not in PRODUCTS]
    if bad:
        print(f"unknown product(s): {bad}")
        return 2
    ok = True
    for k in keys:
        ok = fetch(k) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
