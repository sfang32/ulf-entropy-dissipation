"""sens_win9.py -- window-length sensitivity on the full nine-station set.

§3.6 originally quoted the Pc5 sign reversal from three stations chosen for
latitude spread (run_sens.sh).  This script rebuilds the same table on all nine
Phase-I stations, which is the version the manuscript reports:

    w600   results/mb_zw_{ST}2015_w600_multiband.csv      (the main-chain runs)
    w1200  results/sens9w/mb_zw_{ST}2015_w1200_multiband.csv
    w2400  results/sens9w/mb_zw_{ST}2015_w2400_multiband.csv
    w3600  results/sens9/mb_zw_{ST}2015_w3600_multiband.csv

Output: results/sens_window9.csv  (station, win_min, band, metric, n, rho)

Run on the server with the anaconda python.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
ST = ["MHT", "MZL", "NAT", "SSL", "MLS", "JFT", "LAT", "PXT", "SYS"]
METRICS = ["SE_full", "WPE", "SampEn", "Tsallis_q1.5"]
BANDS = ["Pc3", "Pc4", "Pc5", "all"]

# window (samples) -> (directory, filename template)
WINDOWS = [
    (600,  RES,      "mb_zw_%s2015_w600_multiband.csv"),
    (1200, RES / "sens9w", "mb_zw_%s2015_w1200_multiband.csv"),
    (2400, RES / "sens9w", "mb_zw_%s2015_w2400_multiband.csv"),
    (3600, RES / "sens9",  "mb_zw_%s2015_w3600_multiband.csv"),
]


def extract(f, st, win_min):
    df = pd.read_csv(f)
    ok = df["E_Pc5"].notna() & df["WPE"].notna()
    s = df[ok]
    if len(s) < 100:
        return []
    out = []
    for b in BANDS:
        col = "E_%s" % b
        if col not in s.columns:
            continue
        m2 = s[col].notna()
        le = np.log10(s.loc[m2, col].clip(1e-12))
        for k in METRICS:
            if k not in s.columns or s.loc[m2, k].notna().sum() < 100:
                continue
            out.append({"station": st, "win_min": win_min, "band": b,
                        "metric": k, "n": int(m2.sum()),
                        "rho": float(s.loc[m2, k].corr(le, method="spearman"))})
    return out


def plot(tab):
    """Fig. 4: network mean per band, plus the Pc5 station-by-station flip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "axes.linewidth": 0.8,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                         "legend.fontsize": 11})

    cols = {"SE_full": "#1f77b4", "WPE": "#d62728", "SampEn": "#2ca02c",
            "Tsallis_q1.5": "#9467bd"}
    wins = [10.0, 20.0, 40.0, 60.0]
    labs = ["10", "20", "40", "60"]

    fig = plt.figure(figsize=(13.2, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 2.2], hspace=0.46, wspace=0.10)
    fig.subplots_adjust(top=0.90, bottom=0.10)
    # sharey is required: only the leftmost panel draws y tick labels, so without
    # it the other three panels showed an unlabelled axis (caught in a figure review)
    ax0 = fig.add_subplot(gs[0, 0])
    axes = [ax0] + [fig.add_subplot(gs[0, i], sharey=ax0) for i in range(1, 4)]
    for j, (ax, b) in enumerate(zip(axes, BANDS)):
        for k, c in cols.items():
            ys = [tab[(tab.band == b) & (tab.metric == k)
                      & (tab.win_min == w)].rho.mean() for w in wins]
            ax.plot(wins, ys, "-o", color=c, lw=1.6, ms=4.5, mfc="white",
                    mec=c, mew=1.2, label=k)
        ax.axhline(0, color="0.55", lw=0.8, ls="--")
        # A LINEAR axis.  On a log axis 10->20 and 20->40 take the same width while
        # 40->60 takes less, which readers take for an unequal spacing error; the four
        # window lengths are better served by their true positions on a linear axis.
        ax.set_xticks(wins)
        ax.set_xticklabels(labs)
        ax.set_xlim(0, 70)
        # x-label only on the bottom row; top row shares the same axis
        ax.set_xlabel("")
        ax.set_title("broadband" if b == "all" else b)
        ax.grid(alpha=0.25, lw=0.4)
        if j == 0:
            ax.set_ylabel(r"$\rho$(metric, $\log E_b$)")
            ax.set_ylim(-0.65, 0.45)
            ax.yaxis.set_tick_params(labelleft=True)
    # add_subplot(sharey=...) does not hide the inner tick labels the way
    # plt.subplots(sharey=True) does, so hide them explicitly: the numbers belong on
    # the left edge of the row only
    for _a in axes[1:]:
        _a.yaxis.set_tick_params(labelleft=True)

    # the station-level view of the one coefficient that reverses
    axb = fig.add_subplot(gs[1, :])
    sub = tab[(tab.band == "Pc5") & (tab.metric == "SE_full")]
    for st in ST:
        s = sub[sub.station == st].sort_values("win_min")
        if s.empty:
            continue
        axb.plot(s.win_min, s.rho, "-", color="0.68", lw=1.0, zorder=2)
    mean = [sub[sub.win_min == w].rho.mean() for w in wins]
    npos = [int((sub[sub.win_min == w].rho > 0).sum()) for w in wins]
    axb.plot(wins, mean, "-o", color="#1f77b4", lw=2.6, ms=6.5, zorder=4,
             label="network mean (9 stations)")
    # label every window, but alternate the side so equal counts (0/9 at 10 and 20
    # min) cannot overprint, and drop the label where it would repeat the previous
    # one at the same place
    # Placed at fixed heights near the top of the panel, not relative to the curve:
    # anchored to the points they sat on the grey station traces at 20 and 40 min.
    for _i, (w, n) in enumerate(zip(wins, npos)):
        axb.annotate("%d/9 positive" % n, (w, 0.965 if _i % 2 == 0 else 0.885),
                     xycoords=("data", "axes fraction"), ha="center", va="center",
                     fontsize=11, color="#1f77b4")
    axb.axhline(0, color="0.4", lw=0.9, ls="--")
    axb.set_xticks(wins)
    axb.set_xticklabels(labs)
    axb.set_xlim(0, 70)
    axb.set_xlabel("entropy window (min; linear scale)")
    axb.set_ylabel(r"$\rho(\mathrm{SE_{full}}, \log E_{Pc5})$")
    axb.set_ylim(-0.55, 0.35)
    axb.grid(alpha=0.25, lw=0.4)
    axb.legend(fontsize=9.5, frameon=True, framealpha=0.92, loc="lower right")

    # One figure-level legend above the panels (no collision with the shared x-label
    # that the bottom panel already carries).
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    out = RES / "sens_window9.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("[fig] -> %s" % out)


def main():
    rows = []
    for win, d, tpl in WINDOWS:
        for st in ST:
            f = d / (tpl % st)
            if not f.exists():
                print("  [warn] missing %s" % f, file=sys.stderr)
                continue
            rows += extract(f, st, win / 60.0)
    if not rows:
        raise SystemExit("no inputs found")
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "sens_window9.csv", index=False)

    print("=== window-length sensitivity, %d stations ===" % tab.station.nunique())
    print()
    for b in BANDS:
        print("--- band %s ---" % b)
        print("%-14s %8s %8s %8s %8s" % ("metric", "10 min", "20 min", "40 min", "60 min"))
        for k in METRICS:
            vals = []
            for win in (10.0, 20.0, 40.0, 60.0):
                sub = tab[(tab.band == b) & (tab.metric == k) & (tab.win_min == win)]
                vals.append(sub.rho.mean() if len(sub) else np.nan)
            print("%-14s %+8.3f %+8.3f %+8.3f %+8.3f" % (k, *vals))
        print()

    print("=== SE_full Pc5 per station (sign flip check) ===")
    for win in (10.0, 20.0, 40.0, 60.0):
        sub = tab[(tab.band == "Pc5") & (tab.metric == "SE_full") & (tab.win_min == win)]
        npos = int((sub.rho > 0).sum())
        print("  %4.0f min  n=%d  mean=%+.3f  positive at %d stations"
              % (win, len(sub), sub.rho.mean(), npos))

    plot(tab)


if __name__ == "__main__":
    main()
