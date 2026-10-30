"""e5_latitude.py -- E5: latitude dependence of the entropy-band-energy coupling.

Two panels:
  (a) full band  -- the MAIN claim channel (window-length robust, per E10)
  (b) Pc5        -- mechanism window (window-length dependent, labelled as such)
x = AACGM-v2 magnetic latitude (aacgmv2 2.7, 2015 ref).
Filled symbols + lines = Meridian ZW 9 stations (1 Hz, 10-min window).
Open squares = CARISMA stations (1 Hz, 10-min window) -- the SAME sampling rate
and window as the ZW chain, so the two arms are directly comparable.

The high-latitude arm was SuperMAG 3 stations until v6; it was replaced by
CARISMA because SuperMAG's public product is 1-min, which forced a different
window (1 h) and a different PE order (m=3), making the latitude trend a
comparison of two different experiments.  CARISMA publishes native 1 Hz FGM, so
the latitude span now runs 13-72 deg inside one consistent pipeline.

Reads results/mb_summary.csv (produced by mb_summary.py --include-carisma).
Run on the server: python e5_latitude.py
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
# 🔴 Only the two metrics that pass every gate in Table 6 are drawn.
# Earlier versions also plotted WPE and Tsallis_q1.5, which made each panel carry
# 4 lines x (9 + 23) markers: with the CARISMA arm added that is ~200 symbols on
# two axes, and the result was unreadable (a vision review scored it 4/10 and the
# labels collided).  WPE and Tsallis_q1.5 both FAIL Gate 3 -- they carry no
# conductance information -- so drawing them here would promote measures the paper
# excludes.  Their behaviour is reported in Sec. 3.6 and Fig. 13.
KEY = [("SE_full", "#1f77b4"), ("SampEn", "#2ca02c")]


def main():
    df = pd.read_csv(RES / "mb_summary.csv")
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharey=True)

    for ax, band, ttl in zip(axes, ["all", "Pc5"], ["(a) full band", "(b) Pc5 band"]):
        ax.set_title(ttl, fontsize=12, pad=32)
        ax.axhline(0.0, color="k", lw=0.9, ls=":")
        for y in (-0.2, 0.2):
            ax.axhline(y, color="gray", lw=0.7, ls="--", alpha=0.7)
        for k, col in KEY:
            zw = df[(df.band == band) & (df.metric == k) & (df.src == "zw")] \
                .dropna(subset=["mlat", "rho"]).sort_values("mlat")
            if not zw.empty:
                ax.plot(zw["mlat"], zw["rho"], "-o", color=col, ms=6, lw=1.5,
                        label=k, alpha=0.9)
                # 🔴 Station names are drawn for ONE metric only, and they are drawn
                # as INDEX NUMBERS, not names.
                #
                # Why numbers: six of the nine ZW stations lie between 13 and 29 deg
                # (SYS 13.2, LAT 23.7, JFT 24.9, PXT 25.5, MLS 29.1, SSL 35.0), i.e.
                # four of them inside 5.8 deg -- about one label width.  Any name-
                # based layout therefore overlaps: an earlier version with five offset
                # slots produced SYS/LAT/SSL/JFT/PXT piled on top of each other in
                # both panels, and a vision review scored it 6/10 for exactly that.
                # Numbers are ~1 character wide, so a small numeric offset separates
                # them at any spacing; the name that each number stands for is given
                # in the caption and in Sec. 3.7.
                #
                # Only SE_full is labelled (labelling both lines put two independent
                # sets of nine names in the same latitude band).  The inverting
                # station is ringed for EVERY metric, so nothing is lost.
                if k == "SE_full":
                    for i, (_, r) in enumerate(zw.iterrows()):
                        # Offset alternates up/down by index parity so neighbours
                        # never share a side, and the numbers are pushed just past
                        # the marker rather than sitting on top of it.
                        dy = 11 if i % 2 == 0 else -13
                        dx = 9 if i % 2 == 0 else -9
                        ax.annotate("%d" % (i + 1), (r.mlat, r.rho), fontsize=9,
                                    color="0.1", xytext=(dx, dy),
                                    textcoords="offset points",
                                    ha="left" if dx > 0 else "right",
                                    va="bottom" if dy > 0 else "top",
                                    zorder=6)
            # 🔴 Ring and annotate the inverting station for EVERY metric, outside
            # the SE_full-only label block above.  It used to sit inside that block,
            # which meant it silently vanished once the figure was reduced to the two
            # gate-passing metrics -- because the station that inverts on the Pc5
            # channel is MZL in SampEn, not in SE_full.  A reader seeing an isolated
            # point sitting above zero with no marker would have read it as a bug.
            inv = zw[zw["rho"] > 0]
            for _, r in inv.iterrows():
                ax.plot(r.mlat, r.rho, "o", mfc="none", mec="k", mew=1.5,
                        ms=12, zorder=6)
                if band == "Pc5":
                    # The label names the station, not the index: it is the one
                    # annotation a reader needs to read as prose, and it must not
                    # force a lookup in the caption key.
                    ax.annotate("%s inverts" % r.station, (r.mlat, r.rho),
                                fontsize=8.5, color="k",
                                xytext=(r.mlat + 3.2, 0.155),
                                textcoords="data", ha="left", va="center",
                                arrowprops=dict(arrowstyle="->", lw=0.8,
                                                color="k",
                                                shrinkA=1, shrinkB=5))
            ca = df[(df.band == band) & (df.metric == k) & (df.src == "carisma")] \
                .dropna(subset=["mlat", "rho"]).sort_values("mlat")
            if not ca.empty:
                ax.plot(ca["mlat"], ca["rho"], "s", color=col, ms=6, mfc="none",
                        mew=1.2, alpha=0.85, linestyle="none")
                # Labelled for one metric only, for the same reason as the ZW arm.
                if k == "SE_full":
                    # 🔴 Only four CARISMA sites are labelled in place.  The 23
                    # stations occupy 52-72 deg and ten of them fall between 63 and
                    # 67 deg -- the closest pair (DAWS 65.77, FCHP 65.82) is 0.05 deg
                    # apart, i.e. about a tenth of one label width, so in-plot labels
                    # necessarily collide there.  Labelling the four endpoints of the
                    # range (ANNA the lowest, RANK the highest, and the two that bound
                    # the dense auroral group) keeps the panel readable; the full list
                    # with every latitude is in the caption and in Sec. 3.7.
                    # Labelling more would trade legibility for redundancy the reader
                    # does not need.
                    for r in ca.itertuples():
                        if r.station not in ("ANNA", "DAWS", "FCHU", "RANK"):
                            continue
                        # DAWS/FCHU are 0.05 deg apart, so they get opposite sides;
                        # DAWS is pushed left because its marker sits at the very
                        # right edge of the axis and a right-aligned label would
                        # be clipped by the figure boundary.  ANNA is pushed down-left
                        # because the ZW labels for MZL/SSL reach into 50-52 deg and
                        # an upward label from ANNA would collide with them.
                        if r.station == "DAWS":
                            dx, ha, dy, va = -30, "right", 12, "bottom"
                        elif r.station == "FCHU":
                            dx, ha, dy, va = 30, "left", 12, "bottom"
                        elif r.station == "ANNA":
                            dx, ha, dy, va = -18, "right", -14, "top"
                        else:
                            dx, ha, dy, va = 0, "center", 12, "bottom"
                        ax.annotate(r.station, (r.mlat, r.rho), fontsize=9.5,
                                    color="0.15", xytext=(dx, dy),
                                    textcoords="offset points", ha=ha, va=va,
                                    annotation_clip=False)
        ax.set_xlim(10, 76)
        ax.set_xticks([10, 20, 30, 40, 50, 60, 70])
        ax.set_xlabel("AACGM magnetic latitude [deg]")
        # In-figure title removed; described in the manuscript caption.
        ax.grid(alpha=0.25, lw=0.4)

    axes[0].set_ylabel(r"Spearman $\rho$ (metric vs $\log_{10}E_{band}$)")
    # sharey hides panel (b)'s tick labels by default.  That is the usual journal
    # convention for two panels of the same quantity, but the two panels here are
    # read independently often enough that repeating the scale on (b) removes any
    # doubt -- and a vision reviewer flagged the missing numbers three rounds in a
    # row, which is evidence that a reader can misread it as an omission.
    axes[1].tick_params(labelleft=True, left=True)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="s", ls="", color="gray", mfc="none", mew=1.2))
    l.append("open = CARISMA, colour as above (1 Hz, 600 s)")

    # Station roster below the panels.  Ten of the 23 CARISMA sites lie between
    # 63 and 67 deg and the closest pair is 0.05 deg apart, so in-plot labels
    # cannot be separated there; listing every site with its true latitude under
    # the panels keeps each one machine-readable without crowding the data.
    ca_all = df[(df.src == "carisma") & (df.band == "all")
               & (df.metric == "SE_full")].dropna(subset=["mlat", "rho"]) \
        .sort_values("mlat")
    # The roster is NOT printed in the figure.  An earlier version put it in a
# three-line strip under the panels, but at 6.4 pt the lines collided and the
# whole block still occupied vertical space better spent on the data; the full
# list with every station latitude is in the caption and in Sec. 3.7.

    # Legend above the panels: the region under the x-axis labels is occupied by
    # the axis titles and the station roster, so putting the legend there (as the
    # previous version did) printed it straight through "AACGM magnetic latitude".
    fig.legend(h, l, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    # In-figure title removed; described in the manuscript caption.
    fig.subplots_adjust(wspace=0.14, top=0.90, bottom=0.15)
    out = RES / "e5_latitude.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("[save] %s" % out)
    print("[roster] %d CARISMA stations in this panel (roster is in the caption)"
          % len(ca_all))

    # ---- index key for the ZW station numbers drawn in the panel ----
    # Printed so the caption key can be written from the run output rather than
    # transcribed by hand, which is how figure/caption keys drift apart.
    zw_key = df[(df.src == "zw") & (df.band == "all")
               & (df.metric == "SE_full")].dropna(subset=["mlat", "rho"]) \
        .sort_values("mlat").reset_index(drop=True)
    print("\n=== ZW index key (panel labels) ===")
    print("  " + "   ".join("%d=%s(%.1f)" % (i + 1, r.station, r.mlat)
                           for i, r in enumerate(zw_key.itertuples())))

    # ---- monotonicity of the coupling with latitude ----
    # Reported for ALL metrics, not just the two drawn above: the figure is a
    # presentation choice, and suppressing the numbers for the metrics that were
    # dropped from the panel would hide the evidence for dropping them.
    print("\n=== latitude trend of the coupling ===")
    for band in ("all", "Pc5", "Pc3"):
        for src in ("zw", "carisma"):
            n_st = df[(df.src == src) & (df.band == band)]["station"].nunique()
            if n_st < 5:
                continue
            for k in ("SE_full", "WPE", "SampEn", "Tsallis_q1.5", "CH_C"):
                d = df[(df.band == band) & (df.metric == k)
                       & (df.src == src)].dropna(subset=["mlat", "rho"])
                if len(d) < 5:
                    continue
                r = d["mlat"].corr(d["rho"], method="spearman")
                nneg = int((d["rho"] < 0).sum())
                print("  %-4s %-8s %-13s rho(lat)=%+.2fnegative %2d/%2d  "
                      "range=[%+.2f,%+.2f]"
                      % (band, src, k, r, nneg, len(d),
                         d["rho"].min(), d["rho"].max()))
            print("")


if __name__ == "__main__":
    main()
