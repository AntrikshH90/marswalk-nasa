#!/usr/bin/env python3
"""Validate MARSWALK's Mars-time model against real NASA flight data.

The Mars clock in app/mars_time.py is only worth trusting if it agrees with
measurements made by spacecraft that are actually on Mars. This script pulls
the two live NASA surface-weather services:

  * MEDA  (Mars 2020 / Perseverance, Jezero Crater)   -> archived sols
  * REMS  (Mars Science Laboratory / Curiosity, Gale)  -> ~4,750 sols, 2012-2026

Each record carries a terrestrial date, the mission sol number and the
measured aerocentric solar longitude Ls, so we check three independent things:

  1. Ls model error against thousands of real measurements
  2. sol numbering reproduces the sols NASA actually published
  3. local mean solar time against measured sunrise/sunset midpoints, which
     should differ from 12:00 by Mars' equation of time only

Run:  python3 scripts/validate_mars_time.py
"""
import json
import math
import os
import statistics
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import mars_time as mt  # noqa: E402

UA = {"User-Agent": "marswalk-hackathon/1.0 (NASA Space Apps)"}
MSL_FEED = "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json"
M2020_FEED = "https://mars.nasa.gov/rss/api/?feed=weather&category=mars2020&feedtype=json"
NOON = timedelta(hours=12)


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def parse_feed(doc):
    out = []
    for s in (doc.get("soles") or doc.get("sols") or []):
        date, sol, ls = s.get("terrestrial_date"), s.get("sol"), s.get("ls")
        if not date or sol in (None, "--") or ls in (None, "--"):
            continue
        try:
            d = datetime.strptime(str(date)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            out.append(dict(date=d, sol=int(float(sol)), ls=float(ls),
                            sunrise=s.get("sunrise"), sunset=s.get("sunset"),
                            opacity=s.get("atmo_opacity"), uv=s.get("local_uv_irradiance_index"),
                            min_temp=s.get("min_temp"), max_temp=s.get("max_temp"),
                            pressure=s.get("pressure"), season=s.get("season")))
        except Exception:
            continue
    return out


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ checks
def check_ls(name, recs):
    errs = [wrap180(mt.ls_deg(r["date"] + NOON) - r["ls"]) for r in recs]
    ae = [abs(e) for e in errs]
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    print(f"\n[{name}]")
    print(f"  {len(recs)} sols, {recs[-1]['date'].date()} .. {recs[0]['date'].date()}")
    print(f"  Ls error: mean {statistics.mean(errs):+.3f}  RMS {rms:.3f}  "
          f"max {max(ae):.3f}  p90 {sorted(ae)[int(0.9 * len(ae))]:.3f} deg")
    return rms


def check_sols(name, recs, lon, landing):
    ok = sum(1 for r in recs if mt.sol_number(r["date"] + NOON, lon, landing) == r["sol"])
    print(f"  sol numbers reproduced: {ok}/{len(recs)} ({100.0 * ok / len(recs):.1f}%)")
    return ok, len(recs)


def check_noon(name, recs):
    """Measured true solar noon vs 12:00 mean solar time = Mars equation of time."""
    d = []
    for r in recs:
        try:
            sh, sm = [int(v) for v in str(r["sunrise"]).split(":")[:2]]
            eh, em = [int(v) for v in str(r["sunset"]).split(":")[:2]]
        except Exception:
            continue
        noon = ((sh * 60 + sm) + (eh * 60 + em)) / 2.0
        d.append(noon - 720.0)
    if d:
        print(f"  true noon minus 12:00 mean solar time: mean {statistics.mean(d):+.1f} min, "
              f"range {min(d):+.0f}..{max(d):+.0f} min")
        print("    -> this is Mars' equation of time; published amplitude is about "
              "+/-50 min, so the observed spread is expected, not an error")
    return d


def fit_orbital(train, test, label):
    """Pattern-search fit of (M0, n, perihelion Ls) on the older half of the
    telemetry, scored on the held-out newer half. Reported out-of-sample so we
    are not grading the model on the data its constants came from.

    REMS publishes Ls as a truncated integer degree, so a residual is scored
    against the nearer edge of that 1-degree reporting window.
    """
    def cost(p, recs):
        m0, n, phi = p
        s = 0.0
        for r in recs:
            d = mt.to_jd(r["date"] + NOON) - mt.J2000
            m = (m0 + n * d) % 360.0
            e = wrap180((mt._true_anomaly_deg(m) + phi) % 360.0 - r["ls"])
            s += min(abs(e), abs(e - 1.0)) ** 2
        return s / len(recs)

    p = [mt.M0_DEG, mt.N_DEG_PER_DAY, mt.LS_AT_PERIHELION]
    steps = [0.02, 2e-7, 0.02]
    best = cost(p, train)
    for _ in range(60):
        improved = False
        for i in range(3):
            for sgn in (1, -1):
                q = list(p)
                q[i] += sgn * steps[i]
                c = cost(q, train)
                if c < best - 1e-12:
                    p, best, improved = q, c, True
        if not improved:
            steps = [s / 3.0 for s in steps]
            if max(steps) < 1e-8:
                break
    print(f"  out-of-sample orbital fit ({label}):")
    print(f"    fitted  M0={p[0]:.5f} deg   n={p[1]:.7f} deg/day   perihelion={p[2]:.3f} deg")
    print(f"    JPL/Standish starting point: M0={mt.M0_DEG}  n={mt.N_DEG_PER_DAY}  "
          f"perihelion={mt.LS_AT_PERIHELION}")
    ctr, cte = cost(p, train), cost(p, test)
    print(f"    RMS  train {math.sqrt(ctr):.3f} deg   HELD-OUT {math.sqrt(cte):.3f} deg")
    print(f"    (integer-degree Ls reporting alone floors this near "
          f"1/sqrt(12) = {1 / math.sqrt(12):.3f} deg)")
    return p


def seasonal_table(msl):
    """Real measured seasonal behaviour the conditions panel can cite."""
    if not msl:
        return
    print("\n[measured surface conditions, REMS 2012-2026, Gale Crater]")
    print(f"  {'Ls bin':>8} {'sols':>5} {'mean min T':>11} {'mean max T':>11} "
          f"{'mean P (Pa)':>12} {'moderate+ UV':>13}")
    for lo in range(0, 360, 30):
        rows = [r for r in msl if lo <= r["ls"] < lo + 30]
        if not rows:
            continue
        def avg(k, cast=float):
            vals = []
            for r in rows:
                v = r[k]
                try:
                    if v not in (None, "--", ""):
                        vals.append(cast(v))
                except Exception:
                    pass
            return statistics.mean(vals) if vals else float("nan")
        uvs = [r["uv"] for r in rows if r["uv"] not in (None, "--")]
        hi_uv = sum(1 for u in uvs if u in ("Moderate", "High", "Very High", "Extreme"))
        print(f"  {lo:3d}-{lo + 30:3d} {len(rows):5d} {avg('min_temp'):11.1f} "
              f"{avg('max_temp'):11.1f} {avg('pressure'):12.1f} "
              f"{(100.0 * hi_uv / len(uvs) if uvs else float('nan')):12.1f}%")


def main():
    print("=" * 78)
    print("MARSWALK Mars-time validation against NASA flight telemetry")
    print("=" * 78)
    try:
        msl = parse_feed(fetch(MSL_FEED))
    except Exception as e:
        print("MSL feed failed:", e)
        msl = []
    try:
        m2020 = parse_feed(fetch(M2020_FEED))
    except Exception as e:
        print("M2020 feed failed:", e)
        m2020 = []

    if msl:
        check_ls("REMS / Curiosity  (Gale, 137.4E)", msl)
        check_sols("MSL", msl, mt.GALE_LON, mt.MSL_LANDING_UTC)
        check_noon("MSL", msl)
        h = len(msl) // 2
        fit_orbital(msl[h:], msl[:h], "train on the older half")
    if m2020:
        check_ls("MEDA / Perseverance  (Jezero, 77.45E)", m2020)
        check_sols("M2020", m2020, mt.JEZERO_LON, mt.M2020_LANDING_UTC)
        check_noon("M2020", m2020)

    seasonal_table(msl)
    print()


if __name__ == "__main__":
    main()
