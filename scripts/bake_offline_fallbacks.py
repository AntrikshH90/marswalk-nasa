#!/usr/bin/env python3
"""Bake the offline fallbacks used when the venue Wi-Fi dies mid-demo.

  * data/fallback_photos/  - real NASA Image Library JPEGs + index.jsonl
  * data/mars_weather_fallback.json - MEDA/REMS snapshot incl. the Ls climatology
  * data/donki_flr_fallback.json    - DONKI flare/SEP/CME/GST snapshot

Everything written here is real NASA data captured ahead of time, so a
disconnected laptop still shows Mars imagery and measured surface weather
rather than empty panels.

Run:  python3 scripts/bake_offline_fallbacks.py
"""
import json
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import nasa_live  # noqa: E402

DATA = os.path.join(ROOT, "data")
PHOTOS = os.path.join(DATA, "fallback_photos")
UA = {"User-Agent": "marswalk-hackathon/1.0 (NASA Space Apps)"}
WANT = 12


def log(*a):
    print(*a, flush=True)


def bake_photos():
    os.makedirs(PHOTOS, exist_ok=True)
    # ask the live service for a good spread of surface imagery
    res = nasa_live.get_photos(limit=40)
    photos = res.get("photos", [])
    log(f"  candidate photos: {len(photos)} (origin {res.get('origin')})")
    index_path = os.path.join(PHOTOS, "index.jsonl")
    written, records = 0, []
    for p in photos:
        if written >= WANT:
            break
        url = p.get("url") or ""
        if not url.startswith("http"):
            continue
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png"):
            continue
        fname = f"{p.get('nasa_id') or written}{ext}"
        dest = os.path.join(PHOTOS, fname)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 5000):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=45) as r:
                    blob = r.read()
                if len(blob) < 5000:
                    continue
                with open(dest, "wb") as f:
                    f.write(blob)
            except Exception as e:
                log(f"    skip {fname}: {e}")
                continue
        records.append(dict(file=fname, title=p.get("title", "")[:110],
                            date=p.get("date", ""), center=p.get("center", "NASA"),
                            nasa_id=p.get("nasa_id", ""),
                            description=(p.get("description") or "")[:200]))
        written += 1
        log(f"    [{written:2d}] {fname}  {len(open(dest, 'rb').read()) // 1024} KB")
    with open(index_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    log(f"  wrote {written} bundled images -> {index_path}")


def bake_json(name, obj):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    log(f"  wrote {name} ({os.path.getsize(path) // 1024} KB)")


def main():
    log("[1/3] bundling NASA surface imagery …")
    try:
        bake_photos()
    except Exception as e:
        log(f"  photo bake failed: {e}")

    log("[2/3] snapshotting measured Mars weather …")
    try:
        wx = nasa_live.mars_weather()
        bake_json("mars_weather_fallback.json", wx)
        log(f"    origin {wx.get('origin')}, {wx.get('gale_sols', 0)} REMS sols, "
            f"{len(wx.get('climatology') or [])} Ls bins")
    except Exception as e:
        log(f"  weather snapshot failed: {e}")

    log("[3/3] snapshotting space weather …")
    try:
        sw = nasa_live.space_weather()
        flares = [dict(flrID=r["id"], classType=r["class_type"], beginTime=r["begin"],
                       peakTime=r["peak"], sourceLocation=r["source"], note=r["note"])
                  for r in sw.get("recent", [])]
        if flares:
            bake_json("donki_flr_fallback.json", flares)
        log(f"    {len(flares)} flares cached, advisory {sw['advisory']['level']}")
    except Exception as e:
        log(f"  space-weather snapshot failed: {e}")

    log("done.")


if __name__ == "__main__":
    main()
