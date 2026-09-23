"""Live NASA data for MARSWALK.

Four real feeds, each cached to disk with an offline fallback so the demo
survives a dead venue Wi-Fi:

  * NASA Image & Video Library  (images-api.nasa.gov)  - surface imagery
  * NASA DONKI                  (api.nasa.gov)         - flares, SEP, CME, GST
  * NASA MEDA / Mars 2020       (mars.nasa.gov)        - weather at JEZERO
  * NASA REMS / MSL             (mars.nasa.gov)        - 4,700+ sols, 2012-2026

The API key is read from the NASA_API_KEY environment variable and otherwise
falls back to the key in app/config.py. DEMO_KEY is deliberately not used: it
is rate-limited to ~30 requests/hour, which is not enough for a demo.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from app.config import NASA_API_KEY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "marswalk-hackathon/1.0 (NASA Space Apps Challenge)"}

PHOTO_QUERIES = [
    "Perseverance rover Jezero",
    "Mars 2020 Mastcam-Z Jezero",
    "Jezero crater delta",
    "Curiosity rover Mars surface",
    "Ingenuity Mars helicopter",
]
PHOTO_TTL = 6 * 3600
FLARE_TTL = 1800
WEATHER_TTL = 3600


# ------------------------------------------------------------------ plumbing
def _get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(url, timeout=25):
    return json.loads(_get(url, timeout))


def _load_cache(name, ttl):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        if ttl is None or (time.time() - os.path.getmtime(path)) < ttl:
            try:
                with open(path) as f:
                    return json.load(f), "cache"
            except Exception:
                return None, None
    return None, None


def _load_cache_any(name):
    """Load a cache entry regardless of age (stale beats nothing)."""
    return _load_cache(name, None)


def _save_cache(name, obj):
    try:
        with open(os.path.join(CACHE, name), "w") as f:
            json.dump(obj, f)
    except Exception as e:
        print(f"cache write {name} failed: {e}")


def _bundled(name):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _parse_t(s):
    """DONKI timestamps look like 2026-08-10T12:34Z or ...T12:34:00Z."""
    if not s:
        return None
    s = str(s).strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s[:len(fmt) + 2], fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _donki(path, days, extra=""):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = ("https://api.nasa.gov/DONKI/%s?startDate=%s&endDate=%s&api_key=%s%s"
           % (path, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
              urllib.parse.quote(NASA_API_KEY), extra))
    return _get_json(url, timeout=30)


# ------------------------------------------------------------------- imagery
def _fallback_photos():
    """Bundled real NASA imagery, used when the network is unavailable."""
    out = []
    idx = os.path.join(DATA, "fallback_photos", "index.jsonl")
    if os.path.exists(idx):
        with open(idx) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    out.append(dict(
                        title=rec.get("title", "Mars surface image"),
                        url=f"/fl/{urllib.parse.quote(rec['file'])}",
                        date=rec.get("date", ""), center=rec.get("center", "NASA"),
                        nasa_id=rec.get("nasa_id", ""), description=rec.get("description", ""),
                        source="bundled",
                    ))
                except Exception:
                    pass
    return out


def _fallback_urls(limit=12):
    """Bundled image URLs, offered to the client as swap-ins for any live image
    the NASA CDN refuses (some assets 403 from certain networks)."""
    out = []
    idx = os.path.join(DATA, "fallback_photos", "index.jsonl")
    if os.path.exists(idx):
        with open(idx) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    out.append(f"/fl/{urllib.parse.quote(rec['file'])}")
                except Exception:
                    pass
    return out[:limit]


# ------------------------------------------------------- image proxy + cache
# Hot-linking images-assets.nasa.gov works in a plain browser but fails behind
# client-side network policies, and the venue Wi-Fi may simply not co-operate.
# So every remote image is served from our own origin out of a disk cache.
PHOTO_CACHE = os.path.join(DATA, "photo_cache")
ALLOWED_IMAGE_HOSTS = {
    "images-assets.nasa.gov", "images-api.nasa.gov", "photojournal.jpl.nasa.gov",
    "science.nasa.gov", "mars.nasa.gov", "www.nasa.gov", "nasa.gov",
}


def proxy_url(remote, nasa_id=""):
    return "/api/live/image?u=%s&id=%s" % (urllib.parse.quote(remote, safe=""),
                                          urllib.parse.quote(nasa_id or "", safe=""))


def _cached_photo_path(nasa_id, remote):
    ext = os.path.splitext(urllib.parse.urlparse(remote).path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        ext = ".jpg"
    safe = "".join(c for c in (nasa_id or "img") if c.isalnum() or c in "-_") or "img"
    return os.path.join(PHOTO_CACHE, safe + ext)


def fetch_photo_bytes(remote, nasa_id=""):
    """Return (bytes, content_type) for a NASA image, cached on disk.

    Only hosts on ALLOWED_IMAGE_HOSTS are accepted: this endpoint takes a URL
    from the query string, and an unrestricted one would be an open proxy.
    """
    host = (urllib.parse.urlparse(remote).hostname or "").lower()
    if host not in ALLOWED_IMAGE_HOSTS:
        raise ValueError("host not allowed: %s" % host)
    path = _cached_photo_path(nasa_id, remote)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        with open(path, "rb") as f:
            return f.read(), ("image/png" if path.endswith(".png") else "image/jpeg")
    os.makedirs(PHOTO_CACHE, exist_ok=True)
    data = _get(remote, timeout=30)
    if len(data) < 1000:
        raise ValueError("image too small")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    ctype = "image/png" if data[:8].startswith(b"\x89PNG") else "image/jpeg"
    return data, ctype


def _normalize(photos):
    """Route any stored remote URL back through our own image proxy, so caches
    written before the proxy existed keep working."""
    out = []
    for p in photos:
        p = dict(p)
        u = p.get("url") or ""
        if u.startswith("http"):
            p["remote"] = u
            p["url"] = proxy_url(u, p.get("nasa_id", ""))
        out.append(p)
    return out


def get_photos(limit=16):
    fallbacks = _fallback_urls()
    cached, how = _load_cache("photos.json", PHOTO_TTL)
    if cached is not None:
        cached = _normalize(cached)
        return dict(photos=cached[:limit], origin=how, count=len(cached),
                    fallbacks=fallbacks)
    photos, seen = [], set()
    try:
        for q in PHOTO_QUERIES:
            url = ("https://images-api.nasa.gov/search?media_type=image&q="
                   + urllib.parse.quote(q))
            data = _get_json(url, timeout=25)
            for item in data.get("collection", {}).get("items", [])[:10]:
                d = (item.get("data") or [{}])[0]
                links = item.get("links") or []
                if not links:
                    continue
                nid = d.get("nasa_id", "")
                if not nid or nid in seen:
                    continue
                seen.add(nid)
                href = links[0].get("href", "")
                href = href.replace("~thumb", "~medium").replace("~small", "~medium")
                photos.append(dict(
                    title=(d.get("title") or "")[:120],
                    url=(proxy_url(href, nid) if href.startswith("http") else href),
                    remote=href,
                    date=(d.get("date_created") or "")[:10],
                    center=d.get("center", "NASA"), nasa_id=nid,
                    description=(d.get("description") or "")[:240], source="live",
                ))
                if len(photos) >= 24:
                    break
            if len(photos) >= 24:
                break
    except Exception as e:
        print(f"photos live fetch failed: {e}")
    if photos:
        _save_cache("photos.json", photos)
        return dict(photos=photos[:limit], origin="live", count=len(photos),
                    fallbacks=fallbacks)
    # stale cache, then bundled files
    stale, _ = _load_cache_any("photos.json")
    if stale:
        return dict(photos=stale[:limit], origin="cache-stale", count=len(stale),
                    fallbacks=fallbacks)
    fb = _fallback_photos()
    return dict(photos=fb[:limit], origin="bundled", count=len(fb), fallbacks=fallbacks)


# ------------------------------------------------------------- space weather
FLARE_CLASS_WEIGHT = {"X": 3, "M": 2, "C": 1, "B": 0.4, "A": 0.1}


def _flare_magnitude(class_type):
    """Turn 'M2.4' into a comparable number (log-ish scale)."""
    if not class_type:
        return 0.0
    letter = class_type[0].upper()
    try:
        num = float(class_type[1:])
    except Exception:
        num = 1.0
    return FLARE_CLASS_WEIGHT.get(letter, 0.0) * max(num, 0.1)


def _events():
    """Pull DONKI FLR/SEP/CME/GST, with a bundled snapshot as backup.

    Returns (flares, seps, cmes, gsts, origin).
    """
    cached, how = _load_cache("donki.json", FLARE_TTL)
    if cached is not None:
        return (*cached, how)
    out = []
    ok = False
    try:
        flares = _donki("FLR", 30) or []
        ok = True
    except Exception as e:
        print(f"donki FLR failed: {e}")
        flares = []
    for path, days, key in (("SEP", 30, "seps"), ("CME", 10, "cmes"), ("GST", 14, "gsts")):
        try:
            out.append(_donki(path, days) or [])
            ok = True
        except Exception as e:
            print(f"donki {path} failed: {e}")
            out.append([])
    if ok:
        _save_cache("donki.json", [flares, *out])
        return flares, *out, "live"
    # stale cache first, then the bundled snapshot
    stale, _ = _load_cache_any("donki.json")
    if stale:
        return (*stale, "cache-stale")
    fb = _bundled("donki_flr_fallback.json") or []
    return fb, [], [], [], "bundled"


def space_weather():
    flares_raw, seps_raw, cmes_raw, gsts_raw, origin = _events()
    now = datetime.now(timezone.utc)

    flares = []
    for f in flares_raw:
        t = _parse_t(f.get("beginTime")) or _parse_t(f.get("peakTime"))
        if not t:
            continue
        flares.append(dict(
            id=f.get("flrID", ""), class_type=f.get("classType", "") or "",
            begin=f.get("beginTime", ""), peak=f.get("peakTime", ""),
            source=f.get("sourceLocation", ""), region=f.get("activeRegionNum"),
            note=(f.get("note") or "")[:200], t=t,
            age_h=(now - t).total_seconds() / 3600.0,
        ))
    flares.sort(key=lambda f: f["t"], reverse=True)

    seps = []
    for s in seps_raw:
        t = _parse_t(s.get("eventTime"))
        if not t:
            continue
        seps.append(dict(
            id=s.get("sepID", ""), event_time=s.get("eventTime", ""),
            instruments=[i.get("displayName", "") for i in (s.get("instruments") or [])],
            t=t, age_h=(now - t).total_seconds() / 3600.0,
        ))
    seps.sort(key=lambda s: s["t"], reverse=True)

    cmes = []
    for c in cmes_raw:
        t = _parse_t(c.get("startTime"))
        if not t:
            continue
        speed, half = None, None
        for a in (c.get("cmeAnalyses") or []):
            if a.get("isMostAccurate") or speed is None:
                speed = a.get("speed", speed)
                half = a.get("halfAngle", half)
        cmes.append(dict(id=c.get("activityID", ""), start=c.get("startTime", ""),
                         speed_km_s=speed, half_angle_deg=half, t=t,
                         age_h=(now - t).total_seconds() / 3600.0))
    cmes.sort(key=lambda c: c["t"], reverse=True)

    gsts = []
    for g in gsts_raw:
        t = _parse_t(g.get("startTime"))
        if not t:
            continue
        kps = [k.get("kpIndex") for k in (g.get("allKpIndex") or []) if k.get("kpIndex") is not None]
        gsts.append(dict(id=g.get("gstID", ""), start=g.get("startTime", ""),
                         max_kp=max(kps) if kps else None, t=t,
                         age_h=(now - t).total_seconds() / 3600.0))
    gsts.sort(key=lambda g: g["t"], reverse=True)

    # ---- real radiation assessment for EVA planning -----------------------
    # SEP (solar energetic particles) are the acute hazard for a suited crew;
    # flares are the trigger, CME speed is the driver, and Kp is context.
    sep_72 = [s for s in seps if s["age_h"] <= 72]
    sep_24 = [s for s in seps if s["age_h"] <= 24]
    flr_48 = [f for f in flares if f["age_h"] <= 48]
    biggest_48 = max(flr_48, key=lambda f: _flare_magnitude(f["class_type"])) if flr_48 else None
    fast_cme = [c for c in cmes if (c["speed_km_s"] or 0) >= 800 and c["age_h"] <= 72]
    max_kp_72 = max([g["max_kp"] for g in gsts if g["age_h"] <= 72 and g["max_kp"]] or [0])

    score = 0.0
    reasons = []
    if sep_24:
        score += 55
        reasons.append(f"{len(sep_24)} SEP event(s) within 24 h")
    elif sep_72:
        score += 30
        reasons.append(f"{len(sep_72)} SEP event(s) within 72 h")
    if biggest_48:
        m = _flare_magnitude(biggest_48["class_type"])
        score += min(25.0, 8.0 * m)
        reasons.append(f"largest flare in 48 h: {biggest_48['class_type']}")
    if fast_cme:
        score += 12
        reasons.append(f"{len(fast_cme)} fast CME (>800 km/s) within 72 h")
    score = min(100.0, score)

    if score >= 60:
        level, advice = "WARN", ("EVA radiation risk elevated — restrict to short, "
                                 "shielded traverses and monitor SPE forecasts.")
    elif score >= 25:
        level, advice = "CAUTION", ("Solar particle activity in the past few days — "
                                    "standard EVA radiation protocol, keep transit "
                                    "time to the HAB short.")
    else:
        level, advice = "GO", "No significant solar particle activity — nominal EVA radiation environment."

    advisory = dict(level=level, score=round(score, 1),
                    text=" ".join(reasons) + (" — " + advice if reasons else advice),
                    sep_24h=len(sep_24), sep_72h=len(sep_72),
                    largest_flare_48h=(biggest_48["class_type"] if biggest_48 else None),
                    max_kp_72h=(max_kp_72 or None))

    return dict(
        origin=origin,
        flares_30d=len(flares),
        flares_7d=len([f for f in flares if f["age_h"] <= 24 * 7]),
        seps_30d=len(seps),
        cmes_10d=len(cmes),
        gsts_14d=len(gsts),
        last_flare=(flares[0]["begin"] if flares else None),
        last_sep=(seps[0]["event_time"] if seps else None),
        advisory=advisory,
        recent=[dict(id=f["id"], class_type=f["class_type"], begin=f["begin"],
                     peak=f["peak"], source=f["source"], note=f["note"])
                for f in flares[:8]],
        recent_sep=[dict(id=s["id"], event_time=s["event_time"],
                         instruments=s["instruments"]) for s in seps[:6]],
        recent_cme=[dict(id=c["id"], start=c["start"], speed_km_s=c["speed_km_s"],
                         half_angle_deg=c["half_angle_deg"]) for c in cmes[:6]],
    )


# ---------------------------------------------------------- Mars surface WX
def _weather_feed(category):
    return _get_json("https://mars.nasa.gov/rss/api/?feed=weather&category=%s&feedtype=json"
                     % category, timeout=30)


def _num(v):
    try:
        if v is None or v in ("--", "", "null"):
            return None
        return float(v)
    except Exception:
        return None


def mars_weather():
    """Real surface weather. MEDA reports Jezero itself; REMS (Gale, 2012-2026)
    supplies the long baseline used for the seasonal climatology."""
    cached, how = _load_cache("mars_weather.json", WEATHER_TTL)
    if cached is not None:
        return cached
    out = dict(origin="bundled", jezero=None, gale=None, gale_sols=0,
               climatology=[])
    try:
        m2020 = _weather_feed("mars2020")
        recs = m2020.get("sols") or m2020.get("soles") or []
        if recs:
            latest = recs[-1]
            out["jezero"] = dict(
                sol=int(float(latest.get("sol", 0))),
                terrestrial_date=latest.get("terrestrial_date", ""),
                ls=_num(latest.get("ls")),
                season=latest.get("season", ""),
                min_temp_c=_num(latest.get("min_temp")),
                max_temp_c=_num(latest.get("max_temp")),
                pressure_pa=_num(latest.get("pressure")),
                sunrise=latest.get("sunrise"), sunset=latest.get("sunset"),
                source="NASA MEDA / Mars 2020 (Jezero Crater)",
            )
            out["origin"] = "live"
    except Exception as e:
        print(f"MEDA feed failed: {e}")

    try:
        msl = _weather_feed("msl")
        recs = msl.get("soles") or []
        out["gale_sols"] = len(recs)
        if recs:
            latest = recs[0]
            out["gale"] = dict(
                sol=int(float(latest.get("sol", 0))),
                terrestrial_date=latest.get("terrestrial_date", ""),
                ls=_num(latest.get("ls")), season=latest.get("season", ""),
                min_temp_c=_num(latest.get("min_temp")),
                max_temp_c=_num(latest.get("max_temp")),
                pressure_pa=_num(latest.get("pressure")),
                wind_speed=latest.get("wind_speed"),
                uv_index=latest.get("local_uv_irradiance_index"),
                opacity=latest.get("atmo_opacity"),
                source="NASA REMS / Curiosity (Gale Crater)",
            )
            out["origin"] = "live"
        # seasonal climatology binned by Ls, straight from the measurements
        bins = {}
        for r in recs:
            ls = _num(r.get("ls"))
            if ls is None:
                continue
            b = int(ls // 10) * 10
            d = bins.setdefault(b, dict(n=0, tmin=[], tmax=[], p=[], uv=[]))
            d["n"] += 1
            for k, key in (("tmin", "min_temp"), ("tmax", "max_temp"), ("p", "pressure")):
                v = _num(r.get(key))
                if v is not None:
                    d[k].append(v)
            if r.get("local_uv_irradiance_index") not in (None, "--"):
                d["uv"].append(r["local_uv_irradiance_index"])
        UV_RANK = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4, "Extreme": 5}
        clim = []
        for b in sorted(bins):
            d = bins[b]
            mean = lambda v: (sum(v) / len(v)) if v else None
            uvs = [UV_RANK.get(u, 0) for u in d["uv"]]
            clim.append(dict(
                ls_bin=b, n=d["n"],
                min_temp_c=(round(mean(d["tmin"]), 1) if d["tmin"] else None),
                max_temp_c=(round(mean(d["tmax"]), 1) if d["tmax"] else None),
                pressure_pa=(round(mean(d["p"]), 1) if d["p"] else None),
                uv_mean_rank=(round(mean(uvs), 2) if uvs else None),
            ))
        out["climatology"] = clim
    except Exception as e:
        print(f"REMS feed failed: {e}")

    if out["origin"] == "live":
        _save_cache("mars_weather.json", out)
        return out
    stale, _ = _load_cache_any("mars_weather.json")
    if stale:
        stale["origin"] = "cache-stale"
        return stale
    fb = _bundled("mars_weather_fallback.json")
    if fb:
        return fb
    return out


def climatology_at(ls_deg, clim):
    """Interpolate the measured seasonal table at the current Ls."""
    if not clim:
        return None
    i = int((ls_deg % 360.0) // 10) * 10
    for row in clim:
        if row["ls_bin"] == i:
            return row
    return None


# ------------------------------------------------------------------ dust risk
# Mars global dust storms are a well-documented seasonal phenomenon: the
# classical seasons are Ls ~ 180-260 (perihelion-season, southern summer) for
# regional-to-global storms and a secondary window near Ls ~ 300-330. This is
# climatology from published observations, NOT a live measurement, and the UI
# labels it that way. Measured temperature, pressure and UV from REMS ride
# alongside it so the panel is not purely model-driven.
DUST_SEASON_WEIGHTS = [
    (180, 0.65), (210, 0.95), (240, 1.0), (270, 0.8), (300, 0.55), (330, 0.3),
]


def _dust_season_probability(ls):
    ls = ls % 360.0
    pts = DUST_SEASON_WEIGHTS + [(360 + 180, 0.65)]
    for (a, va), (b, vb) in zip(pts[:-1], pts[1:]):
        if a <= ls <= b:
            t = (ls - a) / max(b - a, 1e-6)
            base = va + t * (vb - va)
            break
    else:
        base = 0.25
    # outside the main seasons the background is lower but non-zero
    if 180 <= ls <= 330:
        return base
    return 0.25


def conditions(clock):
    """Combined EVA conditions panel: measured weather + solar radiation."""
    sw = space_weather()
    wx = mars_weather()
    ls = clock["ls_deg"]

    p_dust = _dust_season_probability(ls)
    if p_dust >= 0.85:
        dust_risk, dust_txt = "HIGH", "Perihelion dust-storm season — expect raised opacity."
    elif p_dust >= 0.55:
        dust_risk, dust_txt = "MODERATE", "Approaching dust-storm season — monitor opacity."
    else:
        dust_risk, dust_txt = "LOW", "Outside the main dust-storm seasons — visibility typically good."

    clim = climatology_at(ls, wx.get("climatology") or [])
    meas = None
    if clim and clim.get("min_temp_c") is not None:
        meas = dict(min_temp_c=clim["min_temp_c"], max_temp_c=clim["max_temp_c"],
                    pressure_pa=clim["pressure_pa"], uv_mean_rank=clim["uv_mean_rank"],
                    n_sols=clim["n"], basis="Gale Crater (REMS), matched by Ls")

    eva_go = dust_risk != "HIGH" and sw["advisory"]["level"] != "WARN"
    if sw["advisory"]["level"] == "WARN":
        hold_reason = "solar particle activity"
    elif dust_risk == "HIGH":
        hold_reason = "dust-storm season"
    else:
        hold_reason = None

    return dict(
        mars=clock,
        dust=dict(risk=dust_risk, text=dust_txt,
                  season_probability=round(p_dust, 2), lsp_deg=ls,
                  basis="published Mars dust-storm climatology"),
        measured=meas,
        at_jezero=wx.get("jezero"),
        at_gale=wx.get("gale"),
        weather_origin=wx.get("origin"),
        gale_sols=wx.get("gale_sols", 0),
        radiation=sw["advisory"],
        space_weather=dict(origin=sw["origin"], flares_30d=sw["flares_30d"],
                           seps_30d=sw["seps_30d"], cmes_10d=sw["cmes_10d"],
                           gsts_14d=sw["gsts_14d"], last_flare=sw["last_flare"],
                           last_sep=sw["last_sep"], recent=sw["recent"],
                           recent_sep=sw["recent_sep"], recent_cme=sw["recent_cme"]),
        eva_go=eva_go,
        hold_reason=hold_reason,
        origin="LIVE" if sw["origin"] in ("live", "cache") else "CACHED",
    )
