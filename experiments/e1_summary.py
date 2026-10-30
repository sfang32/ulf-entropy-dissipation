"""e1_summary.py -- cross-station E1 summary: entropy battery vs Pc5 band energy.

Reads all results/v2_*_pc5energy.csv on the server, computes per-station Spearman
(entropy metric vs logE_pc5 / logP_pc5 / logQ_all) and writes:
  - results/e1_summary.csv      (per-station metric table + AACGM lat)
  - results/e1_corr_vs_mlat.png (correlation vs AACGM magnetic latitude -> E5 prototype)

Run on the server (anaconda3 has pandas/scipy/matplotlib):
  /data/ssd-stor/fsf/anaconda3/bin/python e1_summary.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
# AACGM-v2 (2015-07-01) from v8_entropy_dissipation/data/stations_v8_aacgm.csv
MLAT = {"MHT": 48.36, "MZL": 44.54, "NAT": 38.44, "SSL": 35.04, "MLS": 29.09,
        "PXT": 25.47, "JFT": 24.85, "HZT": 24.35, "LAT": 23.66, "SYT": 21.41,
        "SYS": 13.16}
KEYS = ["SE_full", "SE_Pc5", "PE", "WPE", "SampEn", "Tsallis_q1.5",
        "CH_H", "CH_C", "SVD", "IR_ordinal"]


def load():
    rows, per_station = [], {}
    for f in sorted(glob.glob(str(RES / "v2_*_pc5energy.csv"))):
        st = os.path.basename(f)[3:].split("2015")[0]
        df = pd.read_csv(f)
        ok = df["E_pc5"].notna() & df["WPE"].notna()
        if ok.sum() < 100:
            print("[skip] %s: only %d in-band windows" % (st, ok.sum()))
            continue
        s = df[ok]
        le = np.log10(s["E_pc5"].clip(1e-12))
        lp = np.log10(s["P_pc5"].clip(1e-30))
        lq = np.log10(s["Q_all"].clip(1e-30))
        rec = {"station": st, "mlat": MLAT.get(st, np.nan),
               "n_win": len(df), "n_band": int(ok.sum()),
               "E_pc5_med": s["E_pc5"].median(),
               "P_pc5_med": s["P_pc5"].median()}
        for k in KEYS:
            if k in s.columns and s[k].notna().sum() > 100:
                rec[k + "_vs_logE"] = s[k].corr(le, method="spearman")
                rec[k + "_vs_logP"] = s[k].corr(lp, method="spearman")
                rec[k + "_vs_logQ"] = s[k].corr(lq, method="spearman")
        rows.append(rec)
        per_station[st] = rec
    return pd.DataFrame(rows), per_station


def main():
    tab, per = load()
    if tab.empty:
        raise SystemExit("no results found")
    tab = tab.sort_values("mlat", ascending=False)
    out_csv = RES / "e1_summary.csv"
    tab.to_csv(out_csv, index=False)
    print("[save] %s  (%d stations)" % (out_csv, len(tab)))
    cols = ["station", "mlat", "n_band"] + \
           [c for c in tab.columns if c.endswith("_vs_logE")][:6]
    print(tab[cols].to_string(index=False))

    # ---- E5 prototype: correlation vs AACGM magnetic latitude ----
    metrics = [("SE_full_vs_logE", "SE_full vs logE$_{Pc5}$"),
               ("WPE_vs_logE", "WPE vs logE$_{Pc5}$"),
               ("CH_C_vs_logE", "CH_C vs logE$_{Pc5}$"),
               ("IR_ordinal_vs_logE", "IR$_{ord}$ vs logE$_{Pc5}$")]
    metrics = [(k, lab) for k, lab in metrics if k in tab.columns]
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6),
                             sharey=True)
    for ax, (k, lab) in zip(np.atleast_1d(axes), metrics):
        sub = tab.dropna(subset=[k, "mlat"])
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.scatter(sub["mlat"], sub[k], s=60, c="steelblue", zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["station"], (r["mlat"], r[k]), fontsize=8,
                        xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set_xlabel("AACGM magnetic latitude [deg]")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Spearman $\\rho$")
    fig.suptitle("E1/E5 prototype: entropy-energy correlation vs AACGM latitude", fontsize=12)
    fig.subplots_adjust(wspace=0.12, top=0.84)
    png = RES / "e1_corr_vs_mlat.png"
    fig.savefig(png, dpi=130)
    print("[save] %s" % png)


if __name__ == "__main__":
    main()
