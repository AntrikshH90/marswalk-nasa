#!/usr/bin/env python3
"""
MARSWALK — ML training pipeline
Three models + science-target selection:

  1. traversability  : HistGradientBoostingClassifier  — crew-rated traversability
                       probability from terrain features (trained on EVA
                       engineering-constraint annotations + sensor noise)
  2. science_value   : HistGradientBoostingRegressor   — science utility of a
                       site from mineralogy/topography (trained on expert
                       heuristics derived from Jezero geology + noise)
  3. dust_risk       : GradientBoostingRegressor       — seasonal dust opacity
                       (tau) climatology -> EVA dust risk
  4. targets         : non-max-suppressed peaks of science x traversability
                       with k-means diversity -> 12 curated EVA targets
Outputs models/*.joblib + layers/science.npy + layers/trav.npy + targets.json
"""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              GradientBoostingRegressor)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = os.path.join(ROOT, "layers")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

# Exact footprint of the M20 Jezero CTX DEM mosaic (USGS product metadata).
# Must stay in step with scripts/preprocess.py - the two scripts must georeference
# identically or targets land on the wrong ground.
BBOX = dict(w=76.997437, e=78.582563, n=19.290187, s=17.580520)


def log(*a):
    print(*a, flush=True)


def px_to_lonlat(px, py, W, H):
    lon = BBOX["w"] + (px + 0.5) / W * (BBOX["e"] - BBOX["w"])
    lat = BBOX["n"] - (py + 0.5) / H * (BBOX["n"] - BBOX["s"])
    return float(lon), float(lat)


# ---------------------------------------------------------------- features
def build_features(elev, slope, rough, curv, midx, mprob, aspect):
    """Stack feature channels: [slope, rough, elev_n, curv, sin/cos aspect,
    mineral one-hots (7), mineral confidence]."""
    H, W = elev.shape
    en = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
    ar = np.radians(aspect)
    feats = [slope, np.clip(rough, 0, None), en * 100.0, np.clip(curv, -50, 50),
             np.sin(ar), np.cos(ar)]
    n_classes = int(midx.max()) + 1
    for c in range(n_classes):
        feats.append((midx == c).astype(np.float32))
    feats.append(mprob)
    # distance to steep slope (proximity risk) — gaussian of slope>15
    steep = (slope > 15).astype(np.float32)
    prox = ndimage.gaussian_filter(steep, 8)
    feats.append(prox)
    # terrain position index (wetness/paleolake proxy): inverse smooth elev
    tpi = elev - ndimage.uniform_filter(elev, 31)
    feats.append(tpi)
    return np.stack(feats, axis=-1).astype(np.float32)


def sample_idx(H, W, n, rng):
    ys = rng.integers(0, H, n)
    xs = rng.integers(0, W, n)
    return ys, xs


def main():
    rng = np.random.default_rng(42)
    elev = np.load(os.path.join(LAYERS, "elev.npy"))
    slope = np.load(os.path.join(LAYERS, "slope.npy"))
    rough = np.load(os.path.join(LAYERS, "rough.npy"))
    curv = np.load(os.path.join(LAYERS, "curv.npy"))
    aspect = np.load(os.path.join(LAYERS, "aspect.npy"))
    midx = np.load(os.path.join(LAYERS, "mineral_idx.npy"))
    mconf = np.load(os.path.join(LAYERS, "mineral_conf.npy"))
    H, W = elev.shape
    meta = json.load(open(os.path.join(LAYERS, "meta.json")))
    mnames = meta["mineral_classes"]

    log("[features] …")
    feats = build_features(elev, slope, rough, curv, midx, mconf, aspect)
    n_samples = 250_000
    ys, xs = sample_idx(H, W, n_samples, rng)
    X = feats[ys, xs]

    # ------------------------------------------------- 1. traversability
    log("[model 1] traversability classifier …")
    # Expert annotation function: NASA EVA planning constraints
    #  slope>20° effectively untraversable; roughness>6 m (200m window) is
    #  rock-field class hazard; extreme curvature = boulder terraces.
    s, r, cv = slope[ys, xs], rough[ys, xs], curv[ys, xs]
    p_safe = (1 / (1 + np.exp(np.clip((s - 17.0) / 3.5, -30, 30)))
              * (1 / (1 + np.exp(np.clip((r - 5.0) / 1.6, -30, 30))))
              * (1 / (1 + np.exp(np.clip((np.abs(cv) - 24.0) / 8.0, -30, 30)))))
    # aleatoric label noise (sensor/interpretation uncertainty)
    y_trav = (rng.random(n_samples) < np.clip(p_safe, 0.02, 0.98)).astype(np.uint8)
    Xtr, Xte, ytr, yte = train_test_split(X, y_trav, test_size=0.2, random_state=42)
    clf = HistGradientBoostingClassifier(
        max_iter=320, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=0.1, random_state=42)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    # accuracy against the clean expert rule (denoised reference)
    clean = (p_safe >= 0.5).astype(np.uint8)
    clean_acc = float((clf.predict(X) == clean).mean())
    log(f"  val accuracy (noisy labels) = {acc:.4f}  | vs clean rule = {clean_acc:.4f}")
    joblib.dump(clf, os.path.join(MODELS, "traversability.joblib"))

    # full-grid prediction in chunks -> layers/trav.npy
    log("  predicting full grid …")
    trav = np.zeros((H, W), np.float32)
    chunk = 512 * 512
    flat = feats.reshape(-1, feats.shape[-1])
    for i in range(0, flat.shape[0], chunk):
        trav.ravel()[i:i + chunk] = clf.predict_proba(flat[i:i + chunk])[:, 1]
    np.save(os.path.join(LAYERS, "trav.npy"), trav)

    # ------------------------------------------------- 2. science value
    log("[model 2] science-value regressor …")
    # Expert heuristic: Jezero science objectives (NASA Mars 2020):
    #  biosignature potential in clays/carbonates, sampling diversity,
    #  access to delta-front stratigraphy, olivine alteration experiments.
    w = {"phyllosilicate": 1.0, "carbonate": 1.0, "olivine": 0.55,
         "pyroxene": 0.25, "silica": 0.85, "sulfate": 0.5, "dune_sand": 0.1}
    mids = midx[ys, xs]
    mineral_val = np.zeros(n_samples, np.float32)
    for c, name in enumerate(mnames):
        mineral_val += (mids == c) * w.get(name, 0.3)
    flat_bonus = 1.0 / (1.0 + slope[ys, xs] / 12.0)          # sampling access
    diversity = mconf[ys, xs]                                   # unit clarity
    strat = 1.0 / (1.0 + np.abs(curv[ys, xs]) / 18.0)           # exposed section?
    lowland = 1.0 - (elev[ys, xs] - elev.min()) / (elev.max() - elev.min() + 1e-6)
    y_sci = (2.6 * mineral_val + 0.7 * flat_bonus + 0.5 * diversity
             + 0.4 * strat + 0.35 * lowland)
    y_sci = y_sci + rng.normal(0, 0.25, n_samples)               # label noise
    Xtr, Xte, ytr, yte = train_test_split(X, y_sci, test_size=0.2, random_state=7)
    reg = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.09, max_leaf_nodes=31,
        min_samples_leaf=40, random_state=7)
    reg.fit(Xtr, ytr)
    r2 = reg.score(Xte, yte)
    log(f"  val R^2 = {r2:.4f}")
    joblib.dump(reg, os.path.join(MODELS, "science.joblib"))

    log("  predicting full grid …")
    sci = np.zeros((H, W), np.float32)
    for i in range(0, flat.shape[0], chunk):
        sci.ravel()[i:i + chunk] = reg.predict(flat[i:i + chunk]).astype(np.float32)
    np.save(os.path.join(LAYERS, "science.npy"), sci)

    # combined objective map = science * traversability
    obj = sci * trav
    np.save(os.path.join(LAYERS, "objective.npy"), obj.astype(np.float32))

    # ------------------------------------------------- 3. dust risk
    log("[model 3] dust opacity climatology …")
    # Seasonal tau climatology approximated from published MSL/Viking records:
    # perihelion (Ls~250) dusty season, aphelion clear.
    ls = np.arange(0, 361, 4, dtype=np.float32)
    tau = (0.45 + 0.35 * np.cos(np.radians(ls - 250))
           + 0.18 * np.cos(np.radians(2 * (ls - 250)))
           + 0.08 * np.sin(np.radians(3 * ls)))
    rng2 = np.random.default_rng(3)
    tau += rng2.normal(0, 0.07, ls.shape)
    # add 4 regional storm spikes (historical style)
    for c, amp in [(255, 1.6), (268, 2.4), (150, 1.1), (300, 0.9)]:
        tau += amp * np.exp(-((ls - c) / 3.2) ** 2)
    X_ls = ls.reshape(-1, 1)
    dust = GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                     random_state=1).fit(X_ls, tau)
    joblib.dump(dust, os.path.join(MODELS, "dust.joblib"))
    log("  dust model fitted")

    # ------------------------------------------------- 4. targets
    log("[targets] selecting EVA science targets …")
    # smooth objective, non-max suppression, k-means diversity
    sm = ndimage.gaussian_filter(obj, 6)
    # exclude truly deadly cells
    sm[slope > 24] = 0
    # adaptive peak-picking until we comfortably exceed 12 candidates
    chosen = []
    min_d2 = 90.0 ** 2
    for q, dd in [(99.55, 90), (99.4, 80), (99.2, 70), (98.9, 60), (98.5, 50)]:
        min_d2 = dd ** 2
        mx = ndimage.maximum_filter(sm, size=64)
        peaks = (sm == mx) & (sm > np.percentile(sm, q))
        py, px = np.where(peaks)
        cand = sorted(zip(sm[py, px], py, px), reverse=True)
        chosen = []
        for sc, y, x in cand:
            if all((y - cy) ** 2 + (x - cx) ** 2 >= min_d2 for _, cy, cx in chosen):
                chosen.append((sc, y, x))
            if len(chosen) >= 36:
                break
        if len(chosen) >= 14:
            break
    log(f"  candidate peaks: {len(chosen)} (d>= {int(min_d2**0.5)} px)")
    # best peak of every mineral unit, kept so the final list can be made to
    # span the geology rather than collapsing onto whichever unit scores best
    per_class_best = {}
    for c in range(len(mnames)):
        mask = (midx == c) & (slope < 20) & (sm > 0)
        if mask.any():
            vals = np.where(mask, sm, -1.0)
            iy, ix = np.unravel_index(int(np.argmax(vals)), vals.shape)
            if vals[iy, ix] > 0:
                per_class_best[c] = (float(vals[iy, ix]), int(iy), int(ix))
                chosen.append(per_class_best[c])
    # dedupe candidates closer than 60 px
    dedup = []
    for cand_t in sorted(chosen, key=lambda t: -t[0]):
        if all((cand_t[1] - d[1]) ** 2 + (cand_t[2] - d[2]) ** 2 >= 60 ** 2
               for d in dedup):
            dedup.append(cand_t)
    chosen = dedup

    # k-means on coordinates for diversity, pick best per cluster
    if len(chosen) > 12:
        pts = np.array([[c[1], c[2]] for c in chosen], float)
        km = KMeans(n_clusters=12, n_init=10, random_state=0).fit(pts)
        final = []
        for k in range(12):
            members = [c for c, lab in zip(chosen, km.labels_) if lab == k]
            if members:
                final.append(max(members, key=lambda t: t[0]))
        chosen = final[:12]

    # Spatial k-means says nothing about geology, so the list could still come
    # back as seven sites in one unit. Trade the weakest redundant member for
    # the best site of any unit that has no target yet.
    covered = {int(midx[y, x]) for _, y, x in chosen}
    for c, cand in sorted(per_class_best.items(), key=lambda kv: -kv[1][0]):
        if c in covered or len(chosen) < 12:
            continue
        counts = {}
        for _, y, x in chosen:
            k = int(midx[y, x])
            counts[k] = counts.get(k, 0) + 1
        droppable = [t for t in chosen if counts[int(midx[t[1], t[2]])] > 1]
        if not droppable:
            break
        drop = min(droppable, key=lambda t: t[0])
        chosen.remove(drop)
        chosen.append(cand)
        covered.add(c)
    chosen.sort(key=lambda t: -t[0])

    templates = {
        "phyllosilicate": [
            ("Clay-Rich Paleolake Margin", "Mg-Fe phyllosilicate exposure — high biosignature preservation potential (drill-grade)."),
            ("Delta-Front Mudstone Bench", "Stratified clay-bearing lake sediment — sample for ancient habitability."),
            ("Inflow Channel Alteration Zone", "Fluid-altered sediment along the N inlet — aqueous history chronology."),
        ],
        "carbonate": [
            ("Carbonate Shoal Outcrop", "Lacustrine carbonate — records pH & atmospheric CO₂ of ancient Mars."),
            ("Marginal Carbonate Ring", "Altered margin unit carbonate — in-situ biosignature target."),
            ("Carbonate-Clay Transition", "Contact between carbonate and clay facies — sampling diversity node."),
        ],
        "olivine": [
            ("Olivine Bedrock Exposure", "Fresh mafic bedrock — mineralogy control sample for alteration studies."),
            ("Olivine-Carbonate Margin Unit", "Nili/Faxa-unit analogue — test carbon-cycle hypotheses."),
        ],
        "pyroxene": [
            ("Pyroxene Basaltic Pavement", "Mafic reference outcrop — baseline for spectral cross-calibration."),
            ("Scoured Bedrock Window", "Deflated surface exposing pyroxene bedrock — rapid contact science."),
        ],
        "silica": [
            ("Silica Sinter Prospect", "Silica-rich deposit — potential hydrothermal biosignature trap."),
            ("High-Silica Lentil", "Acid-altered layer — sample for extreme-environment biosignatures."),
        ],
        "sulfate": [
            ("Sulfate-Cemented Dune Lag", "Sulfate-bearing lag deposit — late-stage aqueous alteration probe."),
            ("Evaporite Patch", "Evaporitic sulfate — endpoint of lake desiccation record."),
        ],
        "dune_sand": [
            ("Active Dune Field Traverse", "Mobile sand — aeolian process science + hazard training data."),
            ("Rippled Sand Sheet", "Grain-size distribution study — rover trafficability calibration."),
        ],
    }

    def bearing(y, x):
        """Compass descriptor of where a site sits inside the mapped area."""
        dy, dx = (y / H - 0.5), (x / W - 0.5)
        ns = "N" if dy < -0.07 else "S" if dy > 0.07 else ""
        ew = "W" if dx < -0.07 else "E" if dx > 0.07 else ""
        return (ns + ew) or "central"

    # Names come from short per-unit template pools, so several sites in one
    # unit would otherwise share a name. Give the first site the plain name,
    # then disambiguate the rest by position and relief.
    provisional = []
    for sc, y, x in chosen[:12]:
        c = int(midx[y, x])
        provisional.append(dict(c=c, y=y, x=x,
                                pool=templates.get(mnames[c], [("Science Station", "—")]),
                                name=None, why=None))
    used_names = set()
    for row in provisional:
        for cand_name, cand_why in row["pool"]:
            if cand_name not in used_names:
                row["name"], row["why"] = cand_name, cand_why
                used_names.add(cand_name)
                break
    for c in range(len(mnames)):
        group = [r for r in provisional if r["name"] is None and r["c"] == c]
        if not group:
            continue
        base, base_why = templates.get(mnames[c], [("Science Station", "—")])[0]
        order = sorted(group, key=lambda r: -elev[r["y"], r["x"]])
        tiers = ["upper", "mid", "lower"]
        for n, row in enumerate(order):
            sector = bearing(row["y"], row["x"])
            if len(order) == 1:
                name = f"{base} — {sector} sector"
            else:
                tier = tiers[n] if n < len(tiers) else f"{tiers[-1]} {n - 1}"
                name = f"{base} — {sector} {tier} bench"
            k = 2
            while name in used_names:
                name = f"{name} {['II', 'III', 'IV'][min(k - 2, 2)]}"
                k += 1
            row["name"], row["why"] = name, base_why
            used_names.add(name)

    targets = []
    for i, row in enumerate(provisional):
        c, y, x = row["c"], row["y"], row["x"]
        lon, lat = px_to_lonlat(x, y, W, H)
        targets.append(dict(
            id=f"T{i + 1:02d}",
            name=row["name"],
            rationale=row["why"],
            mineral=mnames[c],
            px=int(x), py=int(y),
            lon=round(lon, 4), lat=round(lat, 4),
            elev=round(float(elev[y, x]), 1),
            slope=round(float(slope[y, x]), 1),
            science=round(float(sci[y, x]), 2),
            traversability=round(float(trav[y, x]), 3),
            objective=round(float(obj[y, x]), 3),
        ))
    with open(os.path.join(ROOT, "targets.json"), "w") as f:
        json.dump(dict(targets=targets), f, indent=2)
    log(f"  wrote {len(targets)} targets -> targets.json")

    metrics = dict(traversability_acc=round(float(acc), 4),
                   traversability_clean_acc=round(float(clean_acc), 4),
                   science_r2=round(float(r2), 4),
                   n_samples=int(n_samples),
                   n_targets=len(targets))
    with open(os.path.join(MODELS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log("metrics:", metrics)
    log("done.")


if __name__ == "__main__":
    main()
