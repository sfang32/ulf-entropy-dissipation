"""plot_multiband.py -- redraw multiband figures from saved CSV (figure-iteration loop).

Applies the fixes from the Qwen vision review of mb_zw_MHT2015_bands.png (5/10, 小改):
  1. titles state "Spearman rho" explicitly, plus sample size n
  2. LaTeX subscripts for band labels, unified units (no "E_Pc3" plain text)
  3. hexbin instead of overlapping scatter (production = ~40k windows/station)
  4. binned median trend line with IQR band (so weak |rho|~0.35 is visually verifiable)
  5. explicit band ranges + data-source wording in the suptitle (no bare "zw")
Reads any mb_*_multiband.csv; usable for both the 1-Hz ZW runs and the 1-min SuperMAG runs.

Run on the server:
  python plot_multiband.py --csv results/mb_zw_MHT2015_multiband.csv \
      --out results/mb_zw_MHT2015_bands_v2.png [--metric SE_full]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND_LAB = {"Pc3": r"$E_{\rm Pc3}$", "Pc4": r"$E_{\rm Pc4}$",
            "Pc5": r"$E_{\rm Pc5}$", "all": r"$E_{\rm all}$"}
BAND_RANGE = {"Pc3": "23-100 mHz", "Pc4": "6.7-23 mHz",
              "Pc5": "1.6-6.7 mHz", "all": "full band"}
SRC_LAB = {"zw": "Meridian ZW, 1 Hz, 600 s window",
           "sm": "SuperMAG, 1 min, 3600 s window"}


def binned_median(x, y, nbins=10):
    """Median y in equal-count bins of x, with IQR -> journal-style trend line."""
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    cx, cy, lo, hi = [], [], [], []
    for i in range(nbins):
        m = (x >= qs[i]) & (x <= qs[i + 1])
        if m.sum() < 5:
            continue
        cx.append(np.median(x[m]))
        cy.append(np.median(y[m]))
        lo.append(np.quantile(y[m], 0.25))
        hi.append(np.quantile(y[m], 0.75))
    return map(np.asarray, (cx, cy, lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="SE_full")
    ap.add_argument("--bins", type=int, default=45)
    ap.add_argument("--global-grid", action="store_true",
                    help="share one x-range across panels (better cross-band compare)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    src = "sm" if os.path.basename(args.csv).startswith("mb_sm") else "zw"
    station = os.path.basename(args.csv).split("_")[2].split("2015")[0]
    year = 2015
    bands = [b for b in ("Pc3", "Pc4", "Pc5", "all") if f"E_{b}" in df.columns]

    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, len(bands), figsize=(3.6 * len(bands), 3.9),
                             sharey=True)

    gx = None
    if args.global_grid:
        allx = np.concatenate([np.log10(df[f"E_{b}"].clip(1e-12).dropna().to_numpy())
                               for b in bands])
        gx = (np.quantile(allx, 0.001), np.quantile(allx, 0.999))

    for ax, b in zip(np.atleast_1d(axes), bands):
        m = df[f"E_{b}"].notna() & df[args.metric].notna()
        x = np.log10(df.loc[m, f"E_{b}"].clip(1e-12).to_numpy())
        y = df.loc[m, args.metric].to_numpy()
        n = x.size
        if n < 10:
            ax.set_visible(False)
            continue
        rho = pd.Series(y).corr(pd.Series(x), method="spearman")
        hb = ax.hexbin(x, y, gridsize=args.bins, cmap="Blues", mincnt=1,
                       bins="log", linewidths=0.0)
        if gx is not None:
            ax.set_xlim(*gx)
        # journal-style binned median + IQR
        cx, cy, lo, hi = binned_median(x, y)
        if cx.size:
            ax.fill_between(cx, lo, hi, color="#d62728", alpha=0.18, lw=0)
            ax.plot(cx, cy, "-o", color="#d62728", ms=3.5, lw=1.5, zorder=5,
                    label="binned median (IQR)")
        ax.set_xlabel(r"$\log_{10}$ %s [nT$^2$]" % BAND_LAB[b])
        # One shared x tick set across panels: the four bands live on different
        # log10 ranges (Pc3 -3.5..0, broadband -2.5..4.5), so per-panel autoscaling
        # gave each panel a different tick set and made visual comparison awkward.
        # A 2-unit grid spanning the global range is readable in all four.
        if gx is not None:
            lo_x, hi_x = gx
        else:
            lo_x, hi_x = np.quantile(x, 0.001), np.quantile(x, 0.999)
        t0 = np.ceil(lo_x / 2.0) * 2.0
        ax.set_xticks(np.arange(t0, hi_x + 0.1, 2.0))
        ax.set_xlim(lo_x, hi_x)
        # rho and sample size restored to the panel title (with a newline so it
        # never truncates); the caption still carries station/metric/source.
        ax.set_title("%s  (%s)\nρ=%+.2f, N=%d" % (b, BAND_RANGE[b], rho, n))
        ax.grid(alpha=0.25, ls=":", lw=0.4, axis="y")
    axes[0].set_ylabel(args.metric)
    h, l = axes[-1].get_legend_handles_labels()
    # Anchor the legend fully ABOVE the figure box (loc="lower center" at y>1) so it
    # cannot reach down into the two-line panel titles; bbox_inches="tight" then
    # grows the canvas upward to hold it. A previous "upper center" at y=1.02 put
    # the legend body inside 0.9-1.0, i.e. on top of the Pc4/Pc5 titles.
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=4,
               frameon=False, columnspacing=1.6, handletextpad=0.5)
    # In-figure title removed; the manuscript caption gives station / metric / source.
    fig.subplots_adjust(wspace=0.12, top=0.88, bottom=0.16)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[plot] -> %s" % args.out)


if __name__ == "__main__":
    main()
