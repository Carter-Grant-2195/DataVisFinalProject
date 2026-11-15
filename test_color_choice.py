"""Validity tests + visual preview for color_choice.py

Run from the repo folder:
    python test_color_choice.py            # run checks, print sample palettes
    python test_color_choice.py --preview  # also write color_preview.html and open it

The preview shows each sample palette as swatches and as a pie chart, with its stats,
so you can judge the colors by eye.
"""

import argparse
import os
import sys
import webbrowser

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color_choice as cc  # noqa: E402

failures = []


def check(name, cond, detail=None):
    status = "PASS" if cond else "FAIL"
    extra = f"  [{detail}]" if detail is not None else ""
    print(f"{status} {name}{extra}")
    if not cond:
        failures.append(name)


def fmt_stats(stats):
    return {k: (round(v, 2) if v is not None else None) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def test_ciede2000():
    # Published test data from Sharma, Wu & Dalal (2005)
    sharma = [
        ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
        ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
        ((50, 2.8361, -74.0200), (50, 0, -82.7485), 3.4412),
        ((50, 0, 0), (50, -1, 2), 2.3669),
        ((50, 2.4900, -0.0010), (50, -2.4900, 0.0009), 7.1792),
        ((50, 2.5, 0), (73, 25, -18), 27.1492),
        ((50, 2.5, 0), (61, -5, 29), 22.8977),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]
    a = np.array([s[0] for s in sharma])
    b = np.array([s[1] for s in sharma])
    expected = np.array([s[2] for s in sharma])
    got = cc.ciede2000(a, b)
    check("CIEDE2000 matches Sharma test data (10 pairs)",
          np.allclose(got, expected, atol=1e-4), f"max err {np.abs(got - expected).max():.2e}")
    check("CIEDE2000 is symmetric", np.allclose(cc.ciede2000(a, b), cc.ciede2000(b, a)))


def test_conversions():
    refs = {"#ffffff": (100, 0, 0), "#000000": (0, 0, 0), "#ff0000": (53.24, 80.09, 67.20),
            "#00ff00": (87.73, -86.18, 83.18), "#0000ff": (32.30, 79.19, -107.86)}
    for hx, lab in refs.items():
        got = cc.hex_to_lab(hx)
        check(f"hex_to_lab {hx}", np.allclose(got, lab, atol=0.05), np.round(got, 2).tolist())

    rng = np.random.default_rng(0)
    hexes = ["#%02x%02x%02x" % tuple(rng.integers(0, 256, 3)) for _ in range(500)]
    check("hex -> Lab -> hex round trip (500 random)",
          all(cc.lab_to_hex(cc.hex_to_lab(h)) == h for h in hexes))


def test_candidate_space():
    cands = cc.build_candidate_space()
    lin = cc.lab_to_linear_rgb(cands)
    check("candidates are in sRGB gamut", np.all((lin > -1e-6) & (lin < 1 + 1e-6)),
          f"{len(cands)} candidates")
    check("candidates are in default lightness range (25-85)",
          cands[:, 0].min() >= 25 and cands[:, 0].max() <= 85)


def test_palettes():
    ok = True
    for n in range(1, 13):
        p = cc.choose_colors(n, random_seed=n)
        ok &= len(p) == n and len(set(p)) == n
    check("palettes n=1..12 have n unique colors", ok)

    check("same random_seed -> same palette",
          cc.choose_colors(8, random_seed=42) == cc.choose_colors(8, random_seed=42))

    p = cc.choose_colors(5, seed_colors=["#1f77b4", "#ff7f0e"])
    check("seed colors are kept first", p[:2] == ["#1f77b4", "#ff7f0e"], p)

    # Small tolerances below account for rounding colors to hex
    p = cc.choose_colors(8, lightness_range=(40, 70), random_seed=3)
    Ls = [float(cc.hex_to_lab(h)[0]) for h in p]
    check("lightness filter respected (40-70)", min(Ls) >= 39 and max(Ls) <= 71,
          [round(x, 1) for x in Ls])

    p = cc.choose_colors(6, hue_range=(180, 300), random_seed=3)
    hs = [float(cc.lab_to_lch(cc.hex_to_lab(h))[2]) for h in p]
    check("hue filter respected (180-300)", all(177 <= h <= 303 for h in hs),
          [round(x, 1) for x in hs])

    p = cc.choose_colors(6, hue_range=(330, 30), random_seed=3)
    hs = [float(cc.lab_to_lch(cc.hex_to_lab(h))[2]) for h in p]
    check("wrapping hue range respected (330-30)", all(h >= 327 or h <= 33 for h in hs),
          [round(x, 1) for x in hs])


def test_weights():
    pd_only = cc.palette_stats(cc.choose_colors(6, seed_colors=["#1f77b4"],
                                                preference_weight=0))
    pp_only = cc.palette_stats(cc.choose_colors(6, seed_colors=["#1f77b4"],
                                                distance_weight=0))
    check("distance-only gives larger min dE than preference-only",
          pd_only["min_delta_e"] > pp_only["min_delta_e"],
          f"{pd_only['min_delta_e']:.1f} vs {pp_only['min_delta_e']:.1f}")
    check("preference-only gives higher pair preference",
          pp_only["mean_pair_preference"] > pd_only["mean_pair_preference"],
          f"{pp_only['mean_pair_preference']:.2f} vs {pd_only['mean_pair_preference']:.2f}")


def test_color_names():
    expected = {"#ff0000": "red", "#0000ff": "blue", "#ffffff": "white", "#00aa00": "green",
                "#ffa500": "orange", "#808080": "gray"}
    got = {hx: cc.color_name(hx) for hx in expected}
    check("color_name gives sensible names", all(expected[hx] in got[hx] for hx in expected), got)


def test_errors():
    cases = [("n_colors=0", dict(n_colors=0)),
             ("too many seeds", dict(n_colors=1, seed_colors=["#000", "#fff"])),
             ("bad hex", dict(n_colors=3, seed_colors=["#12"])),
             ("more colors than candidates", dict(n_colors=100000))]
    for name, kwargs in cases:
        try:
            cc.choose_colors(**kwargs)
            check(f"raises ValueError on {name}", False)
        except ValueError:
            check(f"raises ValueError on {name}", True)


# ---------------------------------------------------------------------------
# Sample palettes (printed, and optionally previewed in HTML)
# ---------------------------------------------------------------------------
def sample_palettes(n):
    samples = [("Tableau-10 (reference)",
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"][:n])]
    for s in range(5):
        samples.append((f"Default weights, random_seed={s}", cc.choose_colors(n, random_seed=s)))
    samples.append(("Seed #1f77b4, distance only",
                    cc.choose_colors(n, seed_colors=["#1f77b4"], preference_weight=0)))
    samples.append(("Seed #1f77b4, preference only",
                    cc.choose_colors(n, seed_colors=["#1f77b4"], distance_weight=0)))
    samples.append(("Seed #1f77b4, equal weights",
                    cc.choose_colors(n, seed_colors=["#1f77b4"],
                                     distance_weight=1, preference_weight=1)))
    return samples


def _pie_svg(colors, size=160):
    import math
    r = size / 2
    k = len(colors)
    if k == 1:
        return (f'<svg width="{size}" height="{size}"><circle cx="{r}" cy="{r}" r="{r - 2}" '
                f'fill="{colors[0]}"/></svg>')
    parts = []
    for i, c in enumerate(colors):
        a0 = 2 * math.pi * i / k - math.pi / 2
        a1 = 2 * math.pi * (i + 1) / k - math.pi / 2
        x0, y0 = r + (r - 2) * math.cos(a0), r + (r - 2) * math.sin(a0)
        x1, y1 = r + (r - 2) * math.cos(a1), r + (r - 2) * math.sin(a1)
        large = 1 if (a1 - a0) > math.pi else 0
        parts.append(f'<path d="M{r},{r} L{x0:.2f},{y0:.2f} A{r - 2},{r - 2} 0 {large} 1 '
                     f'{x1:.2f},{y1:.2f} Z" fill="{c}" stroke="#fff" stroke-width="1"/>')
    return f'<svg width="{size}" height="{size}">{"".join(parts)}</svg>'


def write_preview(samples, path):
    rows = []
    for name, colors in samples:
        stats = fmt_stats(cc.palette_stats(colors))
        swatches = "".join(
            f'<div class="sw"><div class="chip" style="background:{c}"></div><code>{c}</code><br>{cc.color_name(c)}</div>'
            for c in colors)
        rows.append(f"""
        <section>
          <h2>{name}</h2>
          <div class="row">
            {_pie_svg(colors)}
            <div>
              <div class="swatches">{swatches}</div>
              <p class="stats">min &Delta;E: <b>{stats['min_delta_e']}</b> &middot;
                 mean &Delta;E: <b>{stats['mean_delta_e']}</b> &middot;
                 mean pair preference: <b>{stats['mean_pair_preference']}</b></p>
            </div>
          </div>
        </section>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Color choice preview</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fff; color: #222; }}
 section {{ border-bottom: 1px solid #ddd; padding: 12px 0; }}
 h2 {{ font-size: 16px; margin: 0 0 8px; }}
 .row {{ display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }}
 .swatches {{ display: flex; gap: 8px; flex-wrap: wrap; }}
 .sw {{ text-align: center; font-size: 12px; }}
 .chip {{ width: 56px; height: 40px; border-radius: 4px; border: 1px solid #0002; }}
 .stats {{ font-size: 13px; color: #555; }}
</style></head><body>
<h1>Color choice preview</h1>
<p>Higher min &Delta;E (CIEDE2000) = easier to tell the most similar two slices apart
(about 10+ is clearly distinct). Pair preference is 0&ndash;1, higher = predicted more pleasing.</p>
{''.join(rows)}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true",
                        help="write color_preview.html with swatches + pie charts and open it")
    parser.add_argument("-n", type=int, default=6, help="colors per sample palette (default 6)")
    args = parser.parse_args()

    print("=== Checks ===")
    test_ciede2000()
    test_conversions()
    test_candidate_space()
    test_palettes()
    test_weights()
    test_color_names()
    test_errors()

    print(f"\n=== Sample palettes (n={args.n}) ===")
    samples = sample_palettes(args.n)
    for name, colors in samples:
        named = ", ".join(f"{c} {cc.color_name(c)}" for c in colors)
        print(f"{name}:\n  {named}\n  {fmt_stats(cc.palette_stats(colors))}")

    if args.preview:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_preview.html")
        write_preview(samples, path)
        print(f"\nPreview written to {path}")
        webbrowser.open("file:///" + path.replace(os.sep, "/"))

    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
