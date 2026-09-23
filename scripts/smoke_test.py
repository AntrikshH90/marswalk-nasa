#!/usr/bin/env python3
"""Headless browser smoke test + demo screenshots for MARSWALK.

Boots the UI in Chromium, walks every interactive path, asserts on real page
state, and writes the screenshots used in the slide deck to shots/.

Requires Playwright (not in requirements.txt because it pulls a browser):
    pip install playwright && playwright install chromium

Start the server first, then:
    python3 scripts/smoke_test.py [base_url]
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "shots")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

os.makedirs(OUT, exist_ok=True)
errors = []
failures = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 950})
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
        page.on("console",
                lambda m: errors.append(f"CONSOLE-{m.type}: {m.text}")
                if m.type == "error" else None)

        print(f"\nMARSWALK smoke test against {BASE}\n")
        page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("#map-loading.hidden", timeout=45000)
        time.sleep(2.5)

        print("[boot]")
        check("boot completes (loading overlay cleared)",
              page.eval_on_selector("#map-loading", "e => e.classList.contains('hidden')"))
        check("link indicator is OK", "OK" in page.inner_text("#hud-live"))
        check("Mars clock rendered", "SOL" in page.inner_text("#hud-sol")
              and "Ls" in page.inner_text("#hud-ls")
              and "LMST" in page.inner_text("#hud-lmst"))
        n_targets = page.eval_on_selector_all("#target-list .card", "e => e.length")
        check("science targets listed", n_targets == 12, f"{n_targets} cards")
        names = page.eval_on_selector_all("#target-list .tnm", "e => e.map(x => x.textContent)")
        check("target names are distinct", len(set(names)) == len(names))
        check("route auto-planned", page.inner_text("#st-dist") not in ("—", ""),
              f"{page.inner_text('#st-dist')} km, {page.inner_text('#st-time')} h")

        # the canvas must actually have terrain on it, not just a cleared surface
        stats = page.evaluate("""() => {
            const cv = document.getElementById('map');
            const d = cv.getContext('2d').getImageData(0,0,cv.width,cv.height).data;
            let n=0,s=0,s2=0; const set=new Set();
            for (let i=0;i<d.length;i+=4*53){const l=d[i]*.299+d[i+1]*.587+d[i+2]*.114;
              s+=l;s2+=l*l;n++;set.add(`${d[i]>>3},${d[i+1]>>3},${d[i+2]>>3}`);}
            const m=s/n;
            return {std:Math.sqrt(s2/n-m*m), colors:set.size};
        }""")
        check("map canvas has rendered terrain",
              stats["std"] > 5 and stats["colors"] > 30,
              f"std={stats['std']:.1f}, {stats['colors']} colours")
        page.screenshot(path=os.path.join(OUT, "01_overview.png"))

        print("\n[layer rack]")
        page.click("#base-layers .layer:nth-child(2) .nm")   # CTX Ortho
        time.sleep(1.5)
        check("base layer switches", "Ortho" in page.inner_text("#layer-chip"),
              page.inner_text("#layer-chip"))
        page.screenshot(path=os.path.join(OUT, "02_ortho.png"))
        page.click("#base-layers .layer:nth-child(3) .nm")   # THEMIS
        time.sleep(1.5)

        ov = page.query_selector_all("#overlay-layers .layer input")
        ov[3].click()                                        # ML Traversability
        time.sleep(2.5)
        check("overlay toggles", "Traversability" in page.inner_text("#layer-chip"),
              page.inner_text("#layer-chip"))
        ov[2].click()                                        # Mineral Prospectivity
        time.sleep(2.5)
        check("mineral legend appears",
              page.eval_on_selector_all("#legend .lg", "e => e.length") >= 7)
        page.screenshot(path=os.path.join(OUT, "03_ml_overlay.png"))

        print("\n[inspector + probe]")
        page.click("#target-list .card:nth-child(1)")
        time.sleep(1.2)
        check("inspector populated",
              page.eval_on_selector_all("#inspector-body .bar-row", "e => e.length") == 3)
        page.screenshot(path=os.path.join(OUT, "04_inspector.png"))

        box = page.eval_on_selector(
            "#map", "e => { const r = e.getBoundingClientRect();"
                    "return {x:r.x,y:r.y,w:r.width,h:r.height}; }")
        page.mouse.move(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5)
        time.sleep(1.2)
        coord = page.inner_text("#hud-coord")
        check("terrain probe returns coordinates", "°" in coord, coord)
        check("probe reports a mineral unit", "UNIT" in page.inner_text("#hud-min")
              and not page.inner_text("#hud-min").endswith("—"))

        print("\n[live feed]")
        page.click('.tab[data-tab="live"]')
        time.sleep(9)
        n_ph = page.eval_on_selector_all("#photo-grid .ph", "e => e.length")
        loaded = page.evaluate("""() => [...document.querySelectorAll('#photo-grid .ph img')]
            .filter(i => i.complete && i.naturalWidth > 0).length""")
        check("gallery populated", n_ph >= 8, f"{n_ph} tiles")
        check("gallery images render", loaded >= max(1, n_ph - 2),
              f"{loaded}/{n_ph} loaded")
        check("space-weather rows present",
              page.eval_on_selector_all("#flare-box .flare", "e => e.length") >= 1)
        page.screenshot(path=os.path.join(OUT, "05_live.png"))

        print("\n[replan]")
        page.click('.tab[data-tab="targets"]')
        page.click("#target-list .card:nth-child(3)")
        time.sleep(0.4)
        page.click("#insp-add")
        time.sleep(0.4)
        page.click("#btn-plan")
        time.sleep(4)
        check("replan updates stats", page.inner_text("#st-dist") not in ("—", ""),
              f"{page.inner_text('#st-dist')} km / {page.inner_text('#st-time')} h")
        page.screenshot(path=os.path.join(OUT, "06_replan.png"))

        print("\n[3D]")
        page.click("#btn-3d")
        page.wait_for_selector("#view3d:not(.hidden)", timeout=20000)
        time.sleep(9)
        info = page.inner_text("#v3-info")
        check("3D mesh built", "mesh" in info, info)
        page.screenshot(path=os.path.join(OUT, "07_3d.png"))
        page.click("#v3-close")
        time.sleep(0.6)

        print("\n[waypoint + export]")
        page.click('.tool[data-mode="plan"]')
        before = page.eval_on_selector_all("#legs li", "e => e.length")
        page.mouse.click(box["x"] + box["w"] * 0.55, box["y"] + box["h"] * 0.35)
        time.sleep(0.6)
        check("map click adds a waypoint",
              page.eval_on_selector_all("#legs li", "e => e.length") == before + 1)
        try:
            with page.expect_download(timeout=15000):
                page.click("#btn-export")
            check("EVA plan exports", True)
        except Exception as e:
            check("EVA plan exports", False, str(e)[:70])
        page.screenshot(path=os.path.join(OUT, "08_final.png"))

        browser.close()

    print("\n--- console/page errors ---")
    if errors:
        for e in dict.fromkeys(errors):
            print(" ", e[:240])
    else:
        print("  none")

    print(f"\nscreenshots -> {OUT}")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    if any(e.startswith("PAGEERROR") for e in errors):
        print("page errors present")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
