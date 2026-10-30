"""sens_summary.py -- window-length sensitivity of the entropy-energy coupling.

Reads every mb_zw_*_w{N}_multiband.csv (1-Hz ZW runs at several entropy windows) plus
the --decimate 60 runs (1-min口径, results/dec60/) and plots rho(metric, log E_band)
as a function of the entropy window. The Pc5 panel is the critical one: the sign flips
between 10-min and 1-h windows, so this figure defines the robust claim window.

Run on the server: python sens_summary.py
"""
from __future__ import annotations

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
MLAT = {"MHT": 48.36, "NAT": 38.44, "SYS": 13.16}
METRICS = [("SE_full", "#1f77b4"), ("WPE", "#d62728"), ("SampEn", "#2ca02c"),
           ("Tsallis_q1.5", "#9467bd")]
BANDS = ["Pc3", "Pc4", "Pc5", "all"]


def read_runs():
    """-> DataFrame[station, win_min, fs_tag, band, metric, rho, n]"""
    rows = []
    # --- 1-Hz runs with explicit window in the filename ---
    for f in glob.glob(str(RES / "mb_zw_*_w*_multiband.csv")):
        m = re.search(r"mb_zw_([A-Z]{3})2015_w(\d+)_multiband\.csv$", os.path.basename(f))
        if not m:
            continue
        st, win = m.group(1), int(m.group(2))
        rows += extract(f, st, win / 60.0, "1 Hz")
    # --- 1-min (decimated) runs in results/dec60 ---
    for f in glob.glob(str(RES / "dec60" / "mb_zw_*_multiband.csv")):
        st = os.path.basename(f).split("_")[2].split("2015")[0]
        rows += extract(f, st, 60.0, "1 min")
    return pd.DataFrame(rows)


def extract(f, st, win_min, fs_tag):
    df = pd.read_csv(f)
    ok = df["E_Pc5"].notna() & df["WPE"].notna()
    s = df[ok]
    if len(s) < 100:
        return []
    out = []
    for b in BANDS:
        col = f"E_{b}"
        if col not in s.columns:
            continue
        m2 = s[col].notna()
        le = np.log10(s.loc[m2, col].clip(1e-12))
        for k, _ in METRICS:
            if k not in s.columns:
                continue
            v = s.loc[m2, k]
            if v.notna().sum() < 100:
                continue
            out.append({"station": st, "win_min": win_min, "fs_tag": fs_tag,
                        "band": b, "metric": k, "n": int(m2.sum()),
                        "rho": float(v.corr(le, method="spearman"))})
    return out


def main():
    tab = read_runs()
    if tab.empty:
        raise SystemExit("no sensitivity runs found yet")
    out = RES / "sens_window.csv"
    tab.to_csv(out, index=False)
    print("[save] %s  (%d rows)" % (out, len(tab)))

    stations = sorted(tab["station"].unique())
    print("\n=== rho vs window, by band (metric=SE_full) ===")
    piv = tab[tab.metric == "SE_full"].pivot_table(
        index=["band", "win_min"], columns=["station", "fs_tag"], values="rho",
        aggfunc="mean")
    print(piv.round(3).to_string())

    # ---- figure: rho vs window, one panel per band ----
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, len(BANDS), figsize=(3.6 * len(BANDS), 3.7), sharey=True)
    marks = {"MHT": "o", "NAT": "s", "SYS": "^"}
    for ax, b in zip(np.atleast_1d(axes), BANDS):
        sub = tab[tab.band == b]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.axhline(0, color="k", lw=0.8, ls=":")
        for k, col in METRICS:
            ss = sub[sub.metric == k]
            if ss.empty:
                continue
            for st in stations:
                d1 = ss[(ss.station == st) & (ss.fs_tag == "1 Hz")].sort_values("win_min")
                if not d1.empty:
                    ax.plot(d1["win_min"], d1["rho"], "-", color=col, lw=1.4,
                            marker=marks.get(st, "o"), ms=4, alpha=0.85,
                            label=k if (k not in ax.get_legend_handles_labels()[1]
                                        and st == stations[0]) else None)
                d2 = ss[(ss.station == st) & (ss.fs_tag == "1 min")]
                if not d2.empty:
                    ax.plot(d2["win_min"], d2["rho"], "x", color=col, ms=8, mew=2)
        ax.set_xlabel("entropy window [min]")
        ax.set_title("band: %s" % b, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel(r"Spearman $\rho$ (metric vs $\log_{10} E_{band}$)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Window-length sensitivity (Meridian 2015; lines = 1-Hz runs, "
                 "$\\times$ = 1-min decimated runs)\nmarkers: o MHT(48.4$^\\circ$)  "
                 "s NAT(38.4$^\\circ$)  ^ SYS(13.2$^\\circ$)", fontsize=10)
    fig.subplots_adjust(wspace=0.1, top=0.76)
    fig.savefig(RES / "sens_window.png", dpi=140)
    print("[save] sens_window.png")

    # ---- robust-window verdict for Pc5 ----
    print("\n=== Pc5 channel: sign of rho(SE_full / WPE / SampEn) by window ===")
    p = tab[(tab.band == "Pc5") & (tab.metric.isin(["SE_full", "WPE", "SampEn"]))]
    for w in sorted(p["win_min"].unique()):
        q = p[p.win_min == w]
        line = "  %5.0f min:" % w
        for k in ["SE_full", "WPE", "SampEn"]:
            v = q[q.metric == k]["rho"]
            if len(v):
                nneg = int((v < 0).sum())
                line += "  %s mean=%+.2f (%d/%d neg)" % (k, v.mean(), nneg, len(v))
        print(line)


if __name__ == "__main__":
    main()
