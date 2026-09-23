#!/usr/bin/env python3
"""End-to-end smoke test against a running MARSWALK server on :8765.

Boots the API, exercises the headline endpoints, and prints a one-line
verdict that can be pasted into a submission or reviewer email.
"""
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = r"C:/Users/antriksh/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def _get(p, t=15):
    try:
        with urllib.request.urlopen(f"{BASE}{p}", timeout=t) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return -1, str(e).encode()


def _post(p, body, t=20):
    req = urllib.request.Request(
        f"{BASE}{p}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=t) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return -1, str(e).encode()


def _wait_port(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            try:
                s.connect(("127.0.0.1", port)); return True
            except OSError:
                time.sleep(0.5)
    return False


def main():
    # if something is already on the port, assume it's our server
    already_up = False
    with socket.socket() as s:
        try:
            s.connect(("127.0.0.1", PORT)); already_up = True
        except OSError:
            already_up = False

    proc = None
    if not already_up:
        env = {k: v for k, v in os.environ.items()
               if k.lower() not in ("http_proxy", "https_proxy")}
        log = open(os.path.join(ROOT, "uvicorn.log"), "w")
        proc = subprocess.Popen(
            [PY, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=log, stderr=log,
        )
        if not _wait_port(PORT, 60):
            print("server did not come up; see uvicorn.log", file=sys.stderr)
            return 1

    try:
        s, b = _get("/api/health")
        assert s == 200, b
        health = json.loads(b)
        print(f"health: {health}")

        s, b = _get("/api/rover/traverse")
        assert s == 200, b
        trv = json.loads(b)
        print(f"ground truth: {trv['n_samples']} sols {trv['sol_range']} "
              f"-> {trv['distance_km']:.2f} km of real driving")

        s, b = _get("/api/meta")
        assert s == 200, b
        meta = json.loads(b)
        hab = meta["hab"]["px"]
        targets = meta["targets"]
        print(f"targets: {len(targets)} curated science sites")

        # 1. Plan HAB -> T01 -> T03 (auto demo)
        wps = [dict(y=hab["y"], x=hab["x"])] + [
            dict(y=t["py"], x=t["px"]) for t in targets[:3]
        ]
        t0 = time.time()
        s, b = _post("/api/route",
                      dict(waypoints=wps, max_slope=20, crew=2, prefer_science=True))
        assert s == 200, b
        plan = json.loads(b)
        dt = time.time() - t0
        st = plan["stats"]
        print(f"plan HAB->T01..T03: {st['distance_km']:.2f} km, "
              f"{st['time_h']:.2f} h, max grade {st['max_grade']:.1f} deg, "
              f"O2 {st['o2_kg']:.2f} kg, water {st['water_l']:.1f} L "
              f"({plan['nodes_explored']} nodes, {dt*1000:.0f} ms)")

        # 2. Compare to actual rover track over the full mission
        points = trv["points"]
        sol_idx = {}
        for p in points:
            sol_idx.setdefault(p["sol"], []).append(p)
        sols = sorted(sol_idx)
        start, end = sol_idx[sols[0]][0], sol_idx[sols[-1]][-1]
        drove = [(p["x"], p["y"]) for p in points
                 if p["sol"] >= sols[0] and p["sol"] <= sols[-1]]
        d = sum(math.hypot(drove[i][0] - drove[i-1][0],
                           drove[i][1] - drove[i-1][1])
                for i in range(1, len(drove)))
        drove_km = d * meta["grid"]["px_m"] / 1000.0
        print(f"actual rover: {drove_km:.2f} km between sol {sols[0]} and {sols[-1]}")

        # 3. Conditions badge
        s, b = _get("/api/live/conditions")
        assert s == 200, b
        c = json.loads(b)
        eva = "GO" if c["eva_go"] else "HOLD"
        print(f"eva badge: {eva} (dust {c['dust']['risk']}, rad {c['radiation']['level']})")

        # 4. Tiles
        s, b = _get("/api/tiles/terrain/3/0/0.png")
        assert s == 200 and len(b) > 1000, (s, len(b))
        print(f"tile: terrain/3/0/0.png served ({len(b)} bytes)")

        verdict = (
            f"MARSWALK OK | grid {health['grid']} | "
            f"targets {health['targets']} | "
            f"actual rover {drove_km:.1f} km | "
            f"plan {st['distance_km']:.1f} km @ max {st['max_grade']:.1f} deg | "
            f"EVA {eva}"
        )
        print()
        print(verdict)
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
