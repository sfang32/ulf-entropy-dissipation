#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tsallis_plotonly.py -- redraw Fig.7 from the existing tsallis_scan.csv
(no recomputation of the entropy scan)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/"
           "ground_pc5/v8_entropy_dissipation/results")

def main():
    tab = pd.read_csv(RES / "tsallis_scan.csv")
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
    BINS = sorted(tab["bins"].unique())
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for ax, col, ttl in zip(axes, ("rho_Epc5", "rho_Eall"),
                            (r"vs $\log E_{Pc5}$", r"vs $\log E_{\rm all}$")):
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.axhspan(-0.10, 0.10, color="gray", alpha=0.13, lw=0)
        for b, c in zip(BINS, colors):
            g = tab[tab.bins == b].groupby("q")[col]
            mu = g.mean()
            se = g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))
            ax.errorbar(mu.index, mu.values, yerr=se.values, fmt="-o",
                        color=c, ms=5, lw=1.5, capsize=2.5, elinewidth=0.9,
                        label="bins=%d" % b)
        ax.set_xlabel("Tsallis $q$")
        ax.set_title(ttl)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_ylim(-0.15, 0.55)
    axes[0].set_ylabel(r"Spearman $\rho$ (mean $\pm$ SEM over stations)")
    for a_ in axes:
        a_.tick_params(labelleft=True)
    # Shared figure legend at the bottom; both panels use the same bin curves.
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    # In-figure title removed; described in the manuscript caption.
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RES / "tsallis_scan.png", dpi=300)
    print("[save] tsallis_scan.png")

if __name__ == "__main__":
    main()
