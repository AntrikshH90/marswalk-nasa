"""
MARSWALK route planner.
A* on an 80 m planning grid with a hard slope constraint and science-weighted
cost, plus an EVA stats function. Public API matches the contract consumed by
app/main.py: RoutePlanner(plan_slope, plan_rough, plan_elev, meta) and
stats_for_path(path_work, elev, slope, rough, trav, science, px_m, ...).
"""
import heapq
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = os.path.join(ROOT, "layers")

MIN_EDGE_COST = 1.0  # distance is in 80 m planning cells

CONSUMABLES_BASIS = dict(
    o2_kg_per_crew_hour=0.082,
    water_l_per_crew_hour=0.75,
    power_kwh_per_crew_hour=0.10,
)


class RoutePlanner:
    def __init__(self, plan_slope, plan_rough, plan_elev, full_meta):
        self.slope = plan_slope
        self.rough = plan_rough
        self.elev = plan_elev
        self.H, self.W = plan_slope.shape
        self.plan_factor = int(round(full_meta["W"] / self.W))
        self.px_m = full_meta["px_m"] * self.plan_factor
        # normalised cost fields
        self.slope_n = np.clip(plan_slope / 30.0, 0, 1)
        p98 = float(np.percentile(plan_rough, 98)) + 1e-6
        self.rough_n = np.clip(plan_rough / p98, 0, 1)
        self.last_pops = 0

    # ---------------------------------------------------------- single A*
    def plan(self, wps_plan, max_slope=20.0, science=None):
        """wps_plan: list[(row, col)] in planning grid.

        Returns (path, pops) with path as a list of (row, col), or (None, pops)
        when the endpoints are unreachable under the slope limit.
        """
        H, W = self.H, self.W
        start, goal = wps_plan[0], wps_plan[-1]
        blocked = self.slope > max_slope
        blocked[start[0], start[1]] = False
        blocked[goal[0], goal[1]] = False

        def h(r, c):
            dr, dc = abs(r - goal[0]), abs(c - goal[1])
            octile = (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)
            return octile * MIN_EDGE_COST

        def cost(r1, c1, r2, c2):
            base = math.hypot(r2 - r1, c2 - c1)
            pen = (1.0 + 2.6 * self.slope_n[r2, c2] + 1.4 * self.rough_n[r2, c2])
            if science is not None:
                pen *= 1.0 - 0.35 * float(np.clip(science[r2, c2], 0, 1))
            return base * pen

        sr, sc = start
        gr, gc = goal
        openh = [(h(sr, sc), 0.0, sr, sc)]
        g = {(sr, sc): 0.0}
        parent = {}
        seen = set()
        pops = 0
        while openh:
            f, gcost, r, c = heapq.heappop(openh)
            if (r, c) in seen:
                continue
            seen.add((r, c))
            pops += 1
            if (r, c) == (gr, gc):
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= H or nc >= W:
                    continue
                if blocked[nr, nc]:
                    continue          # hard constraint: no crossing, at any price
                ng = gcost + cost(r, c, nr, nc)
                key = (nr, nc)
                if key not in g or ng < g[key]:
                    g[key] = ng
                    parent[key] = (r, c)
                    heapq.heappush(openh, (ng + h(nr, nc), ng, nr, nc))
        self.last_pops = pops
        if (gr, gc) != (sr, sc) and (gr, gc) not in parent:
            return None, pops
        path = [(gr, gc)]
        while path[-1] != (sr, sc):
            path.append(parent[path[-1]])
        path.reverse()
        return path, pops

    # ---------------------------------------------------------- multi-stop
    def plan_multi(self, waypoints, max_slope=20.0, science=None):
        """Chain plan() across a list of waypoints.

        Returns (path, error, notes). path is a list of (row, col) tuples in
        planning-grid coordinates; on infeasibility path is None and error is a
        human-readable string.
        """
        if not waypoints or len(waypoints) < 2:
            return None, "need at least 2 waypoints", []
        full_path = []
        notes = []
        for i in range(len(waypoints) - 1):
            seg, pops = self.plan([waypoints[i], waypoints[i + 1]],
                                  max_slope=max_slope, science=science)
            if seg is None:
                return None, f"infeasible segment {i}->{i + 1} (impassable terrain)", notes
            if i > 0:
                # avoid duplicating the join node
                seg = seg[1:]
            full_path.extend(seg)
        self.last_pops = pops
        return full_path, "", notes


# ---------------------------------------------------------------- stats
def stats_for_path(path_work, elev, slope, rough, trav, science, px_m,
                   n_crew=2, n_stops=0, targets_by_px=None):
    """path_work: (N,2) array of (row, col) in the working grid (40 m)."""
    if path_work is None or len(path_work) < 2:
        return None
    pts = path_work[::max(1, len(path_work) // 4000)]
    if len(pts) < 2:
        pts = path_work
    rows, cols = pts[:, 0], pts[:, 1]
    z = elev[rows, cols]
    dr = np.diff(rows.astype(np.float64))
    dc = np.diff(cols.astype(np.float64))
    seg = np.hypot(dr, dc) * px_m
    dz = np.diff(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.degrees(np.arctan2(np.abs(dz), seg))
    total = float(seg.sum())
    # Speed model (m/s): suited crew on Martian regolith. 1.25 m/s on gentle
    # ground, falling off exponentially with grade, floored at 0.22 m/s.
    speed = 1.25 * np.exp(-np.maximum(grade - 3.0, 0) / 20.0)
    speed = np.clip(speed, 0.22, 1.3)
    move_h = float((seg / speed).sum() / 3600.0)
    stop_h = 0.45 * n_stops
    time_h = move_h + stop_h
    ascent = float(dz[dz > 0].sum())
    descent = float(-dz[dz < 0].sum())
    mean_slope_along = float(np.nanmean(grade)) if len(grade) else 0.0
    max_slope_along = float(np.nanmax(grade)) if len(grade) else 0.0
    b = CONSUMABLES_BASIS
    o2_kg = n_crew * time_h * b["o2_kg_per_crew_hour"]
    water_l = n_crew * time_h * b["water_l_per_crew_hour"]
    power_kwh = n_crew * time_h * b["power_kwh_per_crew_hour"]
    mean_trav = float(np.nanmean(trav[rows, cols])) if len(rows) else 0.0
    mean_sci = float(np.nanmean(science[rows, cols])) if len(rows) else 0.0
    # composable warnings the UI surfaces
    warnings = []
    if max_slope_along > 25:
        warnings.append(f"grade > 25 deg on {int((grade > 25).sum())} segment(s)")
    if mean_trav < 0.45:
        warnings.append("mean traversability low - inspect the slope map")
    return dict(
        distance_m=round(total, 1),
        distance_km=round(total / 1000.0, 3),
        time_h=round(time_h, 2),
        move_h=round(move_h, 2),
        stop_h=round(stop_h, 2),
        mean_grade=round(mean_slope_along, 2),
        max_grade=round(max_slope_along, 2),
        ascent_m=round(ascent, 1),
        descent_m=round(descent, 1),
        o2_kg=round(o2_kg, 2),
        water_l=round(water_l, 2),
        power_kwh=round(power_kwh, 3),
        mean_trav=round(mean_trav, 3),
        mean_science=round(mean_sci, 3),
        warnings=warnings,
        n_crew=n_crew,
        n_stops=n_stops,
    )
