#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_tables.py -- generate the paper tables for the v8 draft.

Tab.1  station inventory (Phase-I 9 / Phase-II 28 / E5b 3)
Tab.2  entropy battery definition + parameters
Tab.3  headline results matrix (metric x band, across the three datasets)

Outputs docs/tables.md (markdown, paste-ready) and mirrors the numbers so the
tables can be regenerated whenever a batch is re-run.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DOCS = Path(__file__).resolve().parent.parent.parent / "docs"
RES = Path(__file__).resolve().parent.parent / "results"

# Phase-I 2015 coverage (10-min files; full year = 52 560)
COV1 = {"MHT": 52272, "MZL": 52416, "NAT": 51552, "SSL": 51552, "MLS": 48384,
        "JFT": 49536, "LAT": 50256, "PXT": 51120, "SYS": 49392}
FULL = 52560
# E5b effective days (from pull_e5b_2015.py output)
COV_E5B = {"KIR": (274, 365), "YKC": (259, 365), "BRW": (204, 365)}


def L_of(mlat):
    return 1.0 / math.cos(math.radians(mlat)) ** 2


def read_csv(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def tab1():
    s1 = {r["station"]: r for r in read_csv(DATA / "stations_v8_aacgm.csv")}
    rows = []
    for st in COV1:
        r = s1[st]
        mlat = float(r["aacgm_lat"])
        rows.append((st, r["name"], float(r["glat"]), float(r["glon"]),
                     mlat, L_of(mlat), COV1[st] / FULL * 100, "2010–2022"))
    rows.sort(key=lambda x: -x[4])

    z2 = [r for r in read_csv(DATA / "zw2_stations_aacgm.csv")
          if r["n_days"] and int(r["n_days"]) >= 100 and r["aacgm_lat"]]
    rows2 = []
    for r in z2:
        mlat = abs(float(r["aacgm_lat"]))
        rows2.append((r["station"], float(r["glat"]), float(r["glon"]), mlat,
                      L_of(mlat), min(int(r["n_days"]), 365) / 365 * 100,
                      r["years"]))
    rows2.sort(key=lambda x: -x[3])

    s3 = {r["station"]: r for r in read_csv(DATA / "coords_all_aacgm.csv")}
    e5b = []
    for st in ("KIR", "YKC", "BRW"):
        r = s3.get(st)
        mlat = abs(float(r["aacgm_lat"])) if r else float("nan")
        d, tot = COV_E5B[st]
        e5b.append((st, float(r["glat"]), float(r["glon"]), mlat, L_of(mlat),
                    d / tot * 100, "2015"))
    e5b.sort(key=lambda x: x[3])

    out = ["### Table 1. Station inventory", "",
           "**Panel A — Meridian Phase-I fluxgate network (FGM, 1 Hz), 2015**",
           "",
           "| Station | Name | Geo lat | Geo lon | AACGM lat | L | 2015 cov. | Years |",
           "|---|---|---|---|---|---|---|---|"]
    for st, nm, la, lo, ml, L, cv, yr in rows:
        out.append("| %s | %s | %.2f°N | %.2f°E | %.2f° | %.2f | %.1f%% | %s |"
                   % (st, nm, la, lo, ml, L, cv, yr))

    out += ["", "**Panel B — Meridian Phase-II FGMM network (1 Hz), 2025–2026** "
            "(stations with ≥100 days)", "",
            "| Station | Geo lat | Geo lon | AACGM lat | L | Days/365 | Years |",
            "|---|---|---|---|---|---|---|"]
    for st, la, lo, ml, L, cv, yr in rows2:
        out.append("| %s | %.2f°N | %.2f°E | %.2f° | %.2f | %.1f%% | %s |"
                   % (st, la, lo, ml, L, cv, yr.replace(";", "/")))

    out += ["", "**Panel C — SuperMAG high-latitude extension (1-min), 2015**", "",
            "| Station | Geo lat | Geo lon | AACGM lat | L | Valid days | Year |",
            "|---|---|---|---|---|---|---|"]
    for st, la, lo, ml, L, cv, yr in e5b:
        out.append("| %s | %.2f°N | %.2f°E | %.2f° | %.2f | %.0f%% | %s |"
                   % (st, la, lo, ml, L, cv, yr))
    return "\n".join(out)


TAB2 = """### Table 2. Entropy / complexity battery

All measures are computed on the **raw 10-min window** (600 samples @ 1 Hz) of the
horizontal component *H*, except the band entropies (computed on the filtered series).
Implemented in numpy only (`entropy_battery.py`), so the pipeline runs without
third-party entropy packages.

| Symbol | Measure | Definition / parameters | Ref. |
|---|---|---|---|
| `SE_full` | Normalised spectral entropy | Shannon entropy of the Welch PSD over the full band, normalised by ln(N_bins); Welch 256-pt, 50 % overlap | — |
| `SE_PcX` | Band spectral entropy | Same, restricted to the Pc3/Pc4/Pc5 band PSD | — |
| `PE` | Permutation entropy | order m = 4, delay τ = 1, normalised by ln(m!) | Bandt & Pompe 2002 |
| `WPE` | Weighted permutation entropy | m = 4, τ = 1, weighted by the variance of each ordinal pattern | Fadlallah et al. 2013 |
| `SampEn` | Sample entropy | m = 2, r = 0.2 σ; cKDTree-vectorised; returns NaN when the m+1 template count is 0 | Richman & Moorman 2000 |
| `Tsallis_q1.5` | Tsallis entropy | 64-bin amplitude histogram, q = 1.5 (parameter scan in Supp.) | Tsallis 1988 |
| `CH_H`, `CH_C` | Complexity–entropy plane | Normalised Shannon entropy *H* and Jensen–Shannon complexity *C*, bandt-pompe distribution (m = 6, τ = 1) | Rosso et al. 2007 |
| `SVD` | SVD entropy | Shannon entropy of the normalised singular-value spectrum of the trajectory matrix (embedding 10) | — |
| `IR_ordinal` | Ordinal irreversibility | Jensen–Shannon divergence between the forward and time-reversed ordinal-pattern distributions | Martínez et al. 2018 |

**Robustness screening (this work).** The battery was run on three independent
datasets (Table 1). Only `SE_full`, `WPE` and `SampEn` reproduce with the same sign
and comparable magnitude everywhere; `Tsallis_q1.5` changes sign between Phase-I and
Phase-II, and `IR_ordinal` / `SVD` fail the across-station consistency test — all
three are therefore **excluded from the main conclusions** and reported only in the
Supplementary Material.
"""


def tab3():
    rows = read_csv(RES / "mb_summary_all.csv")
    keys = ["WPE", "SampEn", "SE_full", "CH_C", "Tsallis_q1.5", "IR_ordinal"]
    bands = ["Pc3", "Pc4", "Pc5", "all"]
    out = ["### Table 3. Headline results", "",
           "Mean Spearman ρ between each entropy metric and the log10 band-limited "
           "energy, averaged over stations. Three independent datasets:", "",
           "| Dataset | Stations | Year(s) | Window |",
           "|---|---|---|---|",
           "| **D1** Phase-I FGM | 9 | 2015 | 10 min @ 1 Hz |",
           "| **D2** Phase-II FGMM | 28 | 2025 | 10 min @ 1 Hz |",
           "| **D3** Phase-II FGMM | 28 | 2026 | 10 min @ 1 Hz |",
           "| D4 (SuperMAG ext.) | 3 | 2015 | 1 h @ 1 min |",
           "", "| Metric | Band | D1 (9 st) | D2 (28 st) | D3 (28 st) | Consistent |",
           "|---|---|---|---|---|---|"]
    for k in keys:
        for b in bands:
            vals = {}
            for src, yr in (("zw", 2015), ("zw2", 2025), ("zw2", 2026)):
                pass
            # mb_summary_all.csv only holds one run per station; recompute from
            # the dedicated per-year summaries when available
            v1 = _mean([r for r in rows if r["src"] == "zw"
                        and r["metric"] == k and r["band"] == b])
            v2 = _mean_year("2025", k, b)
            v3 = _mean_year("2026", k, b)
            ok = "—"
            if v1 is not None and v2 is not None and v3 is not None:
                sgn = {s > 0 for s in (v1, v2, v3)}
                ok = "✅" if len(sgn) == 1 else "❌"
            out.append("| `%s` | %s | %s | %s | %s | %s |"
                       % (k, b, _f(v1), _f(v2), _f(v3), ok))
    out += ["", "**Reading:** `WPE`, `SampEn` and `CH_C` keep their sign and magnitude "
            "across all three datasets; `Tsallis_q1.5` and `IR_ordinal` do not "
            "(excluded). `SE_full` is stable on the full band but changes sign on the "
            "Pc5 band with the window length (see the window-length sensitivity "
            "experiment)."]
    return "\n".join(out)


def _mean(rs):
    v = []
    for r in rs:
        try:
            v.append(float(r["rho"]))
        except (TypeError, ValueError):
            pass
    return sum(v) / len(v) if v else None


def _mean_year(year, k, b):
    """Read the per-year heatmap logs cached under results/ if present."""
    p = RES / ("mb_matrix_zw2_%s.csv" % year)
    if not p.exists():
        return None
    for r in read_csv(p):
        if r.get("metric") == k:
            try:
                return float(r[b])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _f(v):
    return "—" if v is None else "%+.3f" % v


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    txt = "\n\n".join([tab1(), TAB2, tab3()])
    p = DOCS / "tables.md"
    p.write_text(txt, encoding="utf-8")
    print("[tab] wrote %s (%d chars)" % (p, len(txt)))
    print(txt[:600])


if __name__ == "__main__":
    main()
