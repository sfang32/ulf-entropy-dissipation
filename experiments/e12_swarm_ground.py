#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e12_swarm_ground.py -- ground entropy vs INDEPENDENTLY measured Joule heating.

Question (v8 E12): does the ground-field entropy track an ionospheric dissipation
truth value that is NOT derived from the ground series itself?

Truth value: Swarm FAC-derived Joule heating rate Q_J, one value per auroral-zone
crossing (from swarm_timeseries.py). Ground: Meridian 9-station 10-min entropy
battery (mb_zw_*_w600_multiband.csv). Both clocks verified UTC (tz_check.py).

Outputs (results/):
  e12_swarm_ground.csv    metric x {all, quiet, active} Spearman + n
  e12_swarm_ground.png    hexbin(Q_J, WPE) + Q_J-binned medians of the metrics
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
ST = ["MHT", "MZL", "NAT", "SSL", "MLS", "JFT", "LAT", "PXT", "SYS"]
METRICS = ["SE_full", "WPE", "SampEn", "CH_C", "Tsallis_q1.5"]


def ground_table(year=2015, metrics=METRICS):
    """Per-window median over stations for each metric -> DataFrame (time index)."""
    out = {}
    for mt in metrics:
        fr = []
        for st in ST:
            f = RES / ("mb_zw_%s%d_w600_multiband.csv" % (st, year))
            if not f.exists():
                continue
            d = pd.read_csv(f, usecols=lambda c: c in ("time", mt))
            if mt not in d.columns:
                continue
            d["time"] = pd.to_datetime(d["time"])
            fr.append(d.set_index("time")[mt].rename(st))
        if fr:
            out[mt] = pd.concat(fr, axis=1).median(axis=1)
    t = pd.DataFrame(out).sort_index()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossings", default=None)
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--tol-min", type=float, default=30.0)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    out = Path(a.outdir) if a.outdir else RES
    out.mkdir(parents=True, exist_ok=True)
    cs = a.crossings or sorted(glob.glob(str(RES / "swarm_crossings_*.csv")))[-1]
    cr = pd.read_csv(cs)
    cr["t_mid"] = pd.to_datetime(cr["t_mid"])
    cr = cr.sort_values("t_mid")
    gt = ground_table(a.year)
    print("[e12] crossings=%d (%s)  ground windows=%d"
          % (len(cr), Path(cs).name, len(gt)))

    j = pd.merge_asof(cr, gt.sort_index(), left_on="t_mid", right_index=True,
                      direction="nearest",
                      tolerance=pd.Timedelta(minutes=a.tol_min))
    j = j.dropna(subset=["QJ_mean", "WPE"])
    print("[e12] matched crossings = %d" % len(j))

    rows = []
    for mt in [m for m in METRICS if m in j.columns]:
        sub = j.dropna(subset=[mt])
        if len(sub) < 50:
            continue
        r_all = sub["QJ_mean"].corr(sub[mt], method="spearman")
        r_ae = (sub["QJ_mean"].corr(sub["ae"], method="spearman")
                if "ae" in sub else np.nan)
        # split by AE terciles (quiet vs active ionosphere)
        if "ae" in sub and sub["ae"].notna().sum() > 100:
            lo, hi = sub["ae"].quantile([1 / 3, 2 / 3])
            q = sub[sub.ae <= lo]
            ac = sub[sub.ae >= hi]
        else:
            q = ac = sub.iloc[0:0]
        rows.append({
            "metric": mt, "n": len(sub),
            "rho_QJ_all": round(r_all, 4),
            "rho_AE_all": round(r_ae, 4) if np.isfinite(r_ae) else np.nan,
            "n_quiet": len(q),
            "rho_QJ_quiet": round(q["QJ_mean"].corr(q[mt], method="spearman"), 4)
                            if len(q) > 50 else np.nan,
            "n_active": len(ac),
            "rho_QJ_active": round(ac["QJ_mean"].corr(ac[mt], method="spearman"), 4)
                             if len(ac) > 50 else np.nan,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "e12_swarm_ground.csv", index=False)
    print("\n=== E12: Spearman(Q_J per crossing, ground metric) ===")
    print(tab.to_string(index=False))
    sig_txt = ("%.2f" % j["sigma_p"].median()) if "sigma_p" in j.columns else "n/a"
    print("\n[e12] median sigma_p = %s S | QJ_mean med=%.3f p95=%.3f mW/m^2"
          % (sig_txt, j["QJ_mean"].median(), j["QJ_mean"].quantile(0.95)))

    # ---------- figure ----------
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})

    def med_se(v):
        """Standard error of the median ~ 1.2533 * sigma / sqrt(n)."""
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 5:
            return np.nan
        return 1.2533 * np.nanstd(v, ddof=1) / np.sqrt(v.size)

    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.9))
    x = np.log10(j["QJ_mean"].clip(1e-5))
    xlo, xhi = np.quantile(x, [0.01, 0.99])          # trim the sparse left tail
    hb = ax[0].hexbin(x, j["WPE"], gridsize=32, bins="log", cmap="viridis",
                      mincnt=1)
    r = j["QJ_mean"].corr(j["WPE"], method="spearman")
    # binned median trend with 95% CI (Qwen: n=22k needs error bars)
    qs = np.quantile(x, np.linspace(0, 1, 11))
    xm, ym, ye = [], [], []
    for i in range(10):
        m = (x >= qs[i]) & (x <= qs[i + 1])
        if m.sum() > 20:
            xm.append(np.median(x[m]))
            ym.append(np.median(j["WPE"][m]))
            ye.append(1.96 * med_se(j["WPE"][m]))
    ax[0].errorbar(xm, ym, yerr=ye, fmt="-o", color="red", lw=2, ms=5,
                   capsize=3, ecolor="red", elinewidth=1.2,
                   label="binned median $\\pm$95% CI")
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel(r"$\log_{10} \overline{Q_J}$ [mW/m$^2$]")
    ax[0].set_ylabel("ground WPE (median)")
    ax[0].set_xlim(xlo, xhi)
    ax[0].set_ylim(0.12, 1.05)
    cb = fig.colorbar(hb, ax=ax[0], label="count")

    # metric medians by Q_J tercile (+ 95% CI)
    j = j.copy()
    j["QJ_ter"] = pd.qcut(j["QJ_mean"], 3, labels=["low", "mid", "high"])
    ms = [m for m in METRICS if m in j.columns]
    for mt in ms:
        v, e = [], []
        for g in ("low", "mid", "high"):
            col = j.loc[j.QJ_ter == g, mt]
            v.append(np.nanmedian(col))
            e.append(1.96 * med_se(col))
        v = np.asarray(v, dtype=float)
        e = np.asarray(e, dtype=float) / abs(v[0]) if v[0] else e
        ax[1].errorbar([0, 1, 2], v / v[0], yerr=e, fmt="o--", lw=1.6, ms=5,
                       capsize=3, label=mt)
    ax[1].axhline(1.0, color="gray", lw=0.8, ls=":")
    ax[1].set_xticks([0, 1, 2], ["low", "mid", "high"])
    ax[1].set_xlabel(r"$Q_J$ tercile")
    ax[1].set_ylabel("metric median (normalised to low)")
    ax[1].legend(loc="upper left", frameon=False, fontsize=10)
    ax[1].set_ylim(0.60, 1.20)                      # room for the SampEn CI
    for x_ in ax:
        x_.grid(alpha=0.25, lw=0.4)
    png = out / "e12_swarm_ground.png"
    fig.subplots_adjust(wspace=0.30, left=0.075, right=0.985, bottom=0.13, top=0.92)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("[e12] figure -> %s" % png)


if __name__ == "__main__":
    main()
