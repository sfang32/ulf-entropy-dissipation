#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e2_sigma_plotonly.py -- redraw Fig.6 from the existing CSV (no OMNI re-read)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
METRICS = ["SE_full", "WPE", "SampEn", "Tsallis_q1.5", "CH_C", "IR_ordinal"]
SCHEMES = {"const": None, "AE": None, "PC": None}


def main():
    tab = pd.read_csv(RES / "e2_sigma_zw2015.csv")
    sub = tab[tab.band == "Pc5"]
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})

    fig, ax = plt.subplots(1, 2, figsize=(13.8, 5.4))
    order = [m for m in METRICS if m in set(sub["metric"])]
    y = np.arange(len(order))

    cols = {"const": "#7f7f7f", "AE": "#1f77b4", "PC": "#2ca02c"}
    h = 0.26
    for i, sch in enumerate(SCHEMES):
        vals = []
        for k in order:
            r = sub[(sub.scheme == sch) & (sub.metric == k)]["p_P_given_E"].mean()
            vals.append(0.0 if not np.isfinite(r) else r)
        ax[0].barh(y + (i - 1) * h, vals, height=h, color=cols[sch],
                   label=r"$\Sigma_p$: %s" % sch)
    ax[0].axvline(0, color="k", lw=0.9)
    ax[0].set_yticks(y, order)
    ax[0].set_xlabel(r"partial Spearman $\rho$(metric, $\log P_{\mathrm{Pc5}}$ | $\log E_{\mathrm{Pc5}}$)")
    ax[0].set_title("(a) Does the dissipation rate carry information\n"
                    "beyond the band energy?")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30),
                 ncol=3, frameon=False)
    ax[0].grid(alpha=0.22, axis="x", lw=0.4)
    ax[0].set_xlim(-0.25, 0.45)
    ax[0].text(0.5, -0.46,
               r"$const$: $\rho\equiv0$ by construction ($P\propto E$), "
               "which is why that series has no visible bar",
               transform=ax[0].transAxes, ha="center", va="top",
               fontsize=10, color="#555555")

    ae = sub[sub.scheme == "AE"].groupby("metric")[["rho_E", "rho_P"]].mean()
    ae = ae.reindex(order)
    for i, k in enumerate(order):
        ax[1].plot([ae.loc[k, "rho_E"], ae.loc[k, "rho_P"]], [i, i],
                   "-", color="#bbbbbb", lw=1.6, zorder=1)
    ax[1].plot(ae["rho_E"], y, "o", ms=8, color="#7f7f7f",
               label=r"vs $\log E_{\mathrm{Pc5}}$ (energy)")
    ax[1].plot(ae["rho_P"], y, "s", ms=8, color="#d62728",
               label=r"vs $\log P_{\mathrm{Pc5}}$ (dissipation)")
    ax[1].axvline(0, color="k", lw=0.9)
    ax[1].set_yticks(y, order)
    ax[1].set_xlabel(r"Spearman $\rho$")
    ax[1].set_title("(b) With an AE-modulated $\Sigma_P$ the energy\n"
                    "and dissipation axes separate")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30),
                ncol=2, frameon=False)
    ax[1].margins(x=0.14)
    ax[1].grid(alpha=0.22, axis="x", lw=0.4)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, wspace=0.30)
    fig.savefig(RES / "e2_sigma.png", dpi=300, bbox_inches="tight")
    print("[save] e2_sigma.png")


if __name__ == "__main__":
    main()
