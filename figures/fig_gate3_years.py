#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_gate3_years.py -- Fig. 14: the conductance partial correlation, year by year.

Every Phase-I year (2011-2022, 9 stations) and both Phase-II years (2025, 2026,
28 stations).  The point of the figure is the absence of counter-examples: SampEn
and SE_full stay positive in all fourteen data sets, while WPE -- the strongest
raw band coupler -- stays near zero throughout.

Reads the year-tagged per-station CSVs written by e2_sigma.py
(`e2_sigma_{src}{year}.csv`), so the error bars are genuine across-station 95%
confidence intervals rather than a transcribed summary.

Axis note: 2023-2024 were not analysed, so the two phases are placed on a
non-uniform x axis with an explicit break marker.  A categorical axis would have
implied that 2022 and 2025 are one year apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
OUT = RES / "fig_gate3_years.png"

PHASE1 = ([(y, "zw") for y in range(2011, 2023)]
          + [(2023, "zwraw"), (2024, "zwraw")])
PHASE2 = [(y, "zw2") for y in (2025, 2026)]
METRICS = [("SampEn", "#0072B2", "o"), ("SE_full", "#D55E00", "s"),
           ("WPE", "#009E73", "^")]


def year_stats(src, year, metric):
    """(mean, half-95%-CI, n_stations) over stations for the AE scheme, Pc5 band."""
    p = RES / ("e2_sigma_%s%d.csv" % (src, year))
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[(d.band == "Pc5") & (d.scheme == "AE") & (d.metric == metric)]
    v = pd.to_numeric(d["p_P_given_E"], errors="coerce").dropna()
    if v.size < 3:
        return None
    sem = v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(1.96 * sem), int(v.size)


def main():
    rows = []
    for src, group in (("zw", PHASE1), ("zw2", PHASE2)):
        for year, s_ in group:
            rec = {"src": src, "year": year, "phase": "I" if src == "zw" else "II"}
            ok = False
            for nm, _, _ in METRICS:
                st = year_stats(s_, year, nm)
                if st:
                    rec[nm], rec[nm + "_ci"], rec["n"] = st
                    ok = True
            if ok:
                rows.append(rec)
    t = pd.DataFrame(rows)
    print("[fig] %d datasets with per-station CSVs" % len(t))
    if t.empty:
        raise SystemExit("no e2_sigma_*.csv found -- run e2_sigma.py first")

    # 2011-2026 is contiguous now, so the axis is uniform
    t["x"] = list(range(len(t)))
    labels = [str(int(y)) for y in t["year"]]

    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.axhline(0, color="k", lw=0.9, ls=":")
    ax.axhspan(-0.10, 0.10, color="gray", alpha=0.13, lw=0)


    ax.axhline(0.20, color="#999999", lw=0.9, ls="--")
    ax.text(-0.45, 0.207, r"pre-registered $|\rho| \geq 0.2$", ha="left",
            va="bottom", fontsize=9.5, color="#555555")

    for nm, col, mk in METRICS:
        if nm not in t.columns:
            continue
        ax.errorbar(t["x"], t[nm], yerr=t[nm + "_ci"], fmt=mk + "-", color=col,
                    ms=6.5, lw=1.6, capsize=3, elinewidth=0.9, alpha=0.95,
                    label=nm)

    ax.set_xticks(t["x"], labels, rotation=45)
    ax.set_xlabel("data set (year)")
    ax.set_ylabel("partial Spearman " + r"$\rho$" + "(metric, "
                  + r"$\log P_{Pc5}$" + r" $\vert$ " + r"$\log E_{Pc5}$" + ")")
    # No in-figure title; the full description lives in the manuscript caption.
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylim(-0.20, 0.40)
    ax.set_xlim(-0.6, 15.1)
    ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[fig] -> %s" % OUT)
    print("\n=== values used (mean +/- 1.96 SEM across stations) ===")
    for _, r in t.iterrows():
        print("  %s%-4d (n=%d)  %s" % (r.src, r.year, r.n,
              "  ".join("%s=%+.3f+/-%.3f" % (nm, r[nm], r[nm + "_ci"])
                        for nm, _, _ in METRICS if nm in t.columns)))
    t.to_csv(RES / "fig_gate3_years.csv", index=False)
    print("[save] fig_gate3_years.csv")


if __name__ == "__main__":
    main()
