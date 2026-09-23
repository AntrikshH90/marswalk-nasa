#!/usr/bin/env python3
"""Fetch the source Mars 2020 CTX DEM that everything else is built from.

The raster is 90 MB, so it is not kept in the repository; this script pulls it
from the USGS Planetary Maps service (which redirects into the USGS Astrogeology
S3 bucket) and verifies its size and SHA-256 before you use it.

Source product
    M20_JezeroCrater_CTXDEM_20m.tif
    Mars 2020 Science Investigation CTX DEM mosaic, 20 m/px, float32
    4456 x 5067 px, footprint 76.9974-78.5826 E, 17.5805-19.2902 N
    USGS Astrogeology / JPL, delivered in the local "Jezero_Crater_Projection"

Run:  python3 scripts/fetch_data.py          # skips the download if already good
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DEST = os.path.join(DATA, "jezero_dem_20m.tif")

URL = ("https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/"
       "ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXDEM_20m.tif")
EXPECT_BYTES = 90345300
EXPECT_SHA256 = "ba7f706ea1b5d63a562f28a7dcf7a7f3c79bce7f048e6974b57e8a33c30d1ba2"
UA = {"User-Agent": "marswalk-hackathon/1.0 (NASA Space Apps Challenge)"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) == EXPECT_BYTES:
        print(f"already present and the expected size: {DEST}")
        print("sha256:", sha256(DEST))
        return
    print(f"downloading {EXPECT_BYTES / 1e6:.1f} MB …")
    print(f"  {URL}")
    tmp = DEST + ".part"
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if got % (20 << 20) < (1 << 20):
                print(f"  {got / 1e6:.0f} MB", flush=True)
    os.replace(tmp, DEST)
    size = os.path.getsize(DEST)
    if size != EXPECT_BYTES:
        print(f"WARNING: got {size} bytes, expected {EXPECT_BYTES}. "
              "The product may have been re-issued upstream.")
    else:
        print(f"ok: {size} bytes")
    digest = sha256(DEST)
    print("sha256:", digest)
    if EXPECT_SHA256 and digest != EXPECT_SHA256:
        print("ERROR: checksum mismatch - refusing to continue")
        sys.exit(1)
    print("next: python3 scripts/preprocess.py && python3 scripts/train_ml.py")


if __name__ == "__main__":
    main()
