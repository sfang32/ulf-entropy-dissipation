#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_replication.py -- Fig.11: cross-dataset replication of the entropy-band coupling.

Three independent datasets (Table 1):
  D1  Meridian Phase-I  FGM   9 stations  2015      (FGM, 1 Hz)
  D2  Meridian Phase-II FGMM 28 stations  2025      (FGMM01, 1 Hz)
  D3  Meridian Phase-II FGMM 28 stations  2026      (FGMM01, 1 Hz)

D2/D3 share no years, instrument or station set with D1, so agreement is genuine
replication rather than a larger sample of the same data.

Statistics are recomputed from the per-station mb_*_multiband.csv files so that
real error bars (standard error of the mean rho across stations) can be drawn.

Output: results/fig_replication.png
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")

# Okabe-Ito colour-blind-safe palette
DATASETS = [("zw", 2015, "D1  Phase-I FGM, 9 st, 2015", "#0072B2", "o"),
            ("zw2", 2025, "D2  Phase-II FGMM, 28 st, 2025", "#D55E00", "s"),
            ("zw2", 2026, "D3  Phase-II FGMM, 28 st, 2026", "#009E73", "^")]
METRICS = ["WPE", "SampEn", "SE_full", "CH_C", "Tsallis_q1.5", "IR_ordinal"]
BANDS = ["Pc3", "Pc4", "Pc5", "all"]


def station_rhos(src, year, metric, band):
    """Per-station Spearman(metric, log10 E_band) for one dataset.

    IMPORTANT: the window tag is pinned to w600. A looser pattern such as
    "mb_zw_*2015_w*_multiband.csv" also matches the window-sensitivity runs
    (w1200/2400/3600), which would silently mix three extra window lengths into
    the mean for the stations that were part of that experiment.
    """
    pat = str(RES / ("mb_%s_*%d_w600_multiband.csv" % (src, year)))
    out = []
    for f in sorted(glob.glob(pat)):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        col = "E_" + band
        if col not in d.columns or metric not in d.columns:
            continue
        ok = d[col].notna() & d[metric].notna()
        if ok.sum() < 100:
            continue
        r = d.loc[ok, metric].corr(
            np.log10(d.loc[ok, col].clip(1e-12)), method="spearman")
        if np.isfinite(r):
            out.append(r)
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "fig_replication.png"))
    a = ap.parse_args()

    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "legend.fontsize": 11,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharey=True)
    xs = np.arange(len(BANDS))
    for ax, mt in zip(axes.ravel(), METRICS):
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.axhspan(-0.10, 0.10, color="gray", alpha=0.13, lw=0)
        for src, yr, lab, col, mk in DATASETS:
            means, errs = [], []
            for b in BANDS:
                r = station_rhos(src, yr, mt, b)
                means.append(np.mean(r) if r.size else np.nan)
                errs.append(np.std(r, ddof=1) / np.sqrt(r.size)
                            if r.size > 1 else np.nan)
            ax.errorbar(xs, means, yerr=errs, fmt=mk + "-", color=col, ms=8,
                        lw=1.9, capsize=3.5, elinewidth=1.1, alpha=0.95,
                        label=lab)
        ax.set_xticks(xs, BANDS)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_title(mt, pad=9)              # neutral title
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel("band")                # every panel (Qwen #3)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"mean Spearman $\rho$")   # short label; details -> caption
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)   # series only
    # In-figure title removed; the manuscript caption identifies the figure.
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[fig] -> %s" % a.out)
    print("[fig] caption note: error bars = SEM across stations; "
          "grey band = |rho| < 0.10 (no usable signal); panels share the y axis")

    # console summary (for the caption / honesty check)
    print("\nmetric            band   D1        D2        D3")
    for mt in METRICS:
        for b in BANDS:
            row = []
            for src, yr, lab, col, mk in DATASETS:
                r = station_rhos(src, yr, mt, b)
                row.append("%+.3f" % np.mean(r) if r.size else "  n/a ")
            print("%-17s %-6s %s  %s  %s" % (mt, b, row[0], row[1], row[2]))


if __name__ == "__main__":
    main()
