#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""case_study_2015.py -- magnetosphere-ionosphere-ground joint case study.

Event: 2015-03-17 "St. Patrick's Day" storm (SYM-H min ~ -223 nT).

Panels:
  (1) SYM-H [nT]              -- OMNI 1-min
  (2) AE [nT]                 -- OMNI 1-min (auroral electrojet -> Joule proxy)
  (3) Swarm Q_J per polar crossing [mW/m^2]  -- swarm_timeseries output (E12a)
  (4) ground entropy (median over Meridian stations) -- SE_full / WPE

Purpose: test whether the GROUND entropy responds to an INDEPENDENTLY measured
ionospheric dissipation (Swarm Joule heating) on storm time scales.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import cdflib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMNI = Path("/data/ftp/fsf/databases/data_omni/omni_1min")
RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
ZW_STATIONS = ["MHT", "MZL", "NAT", "SSL", "MLS", "JFT", "LAT", "PXT", "SYS"]


def load_omni(t0, t1, vars_=("SYM_H", "AE_INDEX")):
    frames = []
    for f in sorted(glob.glob(str(OMNI / "2015" / "omni_hro_1min_201503*.cdf"))):
        c = cdflib.CDF(f)
        t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
        d = {"time": t}
        for v in vars_:
            try:
                x = np.asarray(c.varget(v), dtype=float)
                d[v] = np.where(np.abs(x) > 1e30, np.nan, x)
            except Exception:
                pass
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames).set_index("time").sort_index()
    return df.loc[t0:t1]


def ground_entropy(t0, t1, stations=ZW_STATIONS, metric="SE_full"):
    """Median over stations of a metric in the Pc5 window (10-min, w600)."""
    frames = []
    for st in stations:
        f = RES / ("mb_zw_%s2015_w600_multiband.csv" % st)
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=lambda c: c in ("time", metric))
        if metric not in d.columns:
            continue
        d["time"] = pd.to_datetime(d["time"])
        d = d[(d.time >= t0) & (d.time <= t1)]
        if d.empty:
            continue
        frames.append(d.set_index("time")[metric].rename(st))
    if not frames:
        return None
    m = pd.concat(frames, axis=1)
    return m.median(axis=1), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-03-15")
    ap.add_argument("--end", default="2015-03-21")
    ap.add_argument("--crossings", default=None,
                    help="swarm_crossings csv (default: auto-detect)")
    ap.add_argument("--metric", default="SE_full")
    ap.add_argument("--band", default="Pc5")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    t0, t1 = pd.Timestamp(a.start), pd.Timestamp(a.end)
    out = Path(a.outdir) if a.outdir else RES
    out.mkdir(parents=True, exist_ok=True)

    om = load_omni(t0, t1)
    csv = a.crossings or sorted(glob.glob(str(RES / "swarm_crossings_*.csv")))[-1]
    cr = pd.read_csv(csv)
    cr["t_mid"] = pd.to_datetime(cr["t_mid"])
    cr = cr[(cr.t_mid >= t0) & (cr.t_mid <= t1)]
    print("[cs] omni %d pts, crossings %d (%s)"
          % (len(om), len(cr), Path(csv).name))

    g = ground_entropy(t0, t1, metric=a.metric)

    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(om.index, om["SYM_H"], color="#333333", lw=1.1)
    ax[0].axhline(-50, color="r", lw=0.9, ls="--",
                  label="storm threshold (SYM-H = $-50$ nT)")
    ax[0].legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax[0].set_ylabel("SYM-H [nT]")
    ax[1].plot(om.index, om["AE_INDEX"], color="#8c564b", lw=0.8)
    ax[1].set_ylabel("AE [nT]")

    for sat, col in (("A", "#1f77b4"), ("B", "#d62728")):
        sub = cr[cr.sat == sat]
        if sub.empty:
            continue
        # NB: clipping the series at 8e-5 stacked every quiet crossing onto a flat
        # floor at the bottom of the log axis, which read as "data cut off".  The
        # clip is now only a guard against exact zeros entering the log, and the
        # axis limit below is set to the actual data range.
        ax[2].plot(sub.t_mid, sub.QJ_mean.clip(lower=1e-9), "-o", ms=4, lw=0.8,
                   color=col, label="Swarm-%s" % sat, alpha=0.85)
    ax[2].set_ylabel(r"$\overline{Q_J}$ per crossing [mW/m$^2$]")
    ax[2].legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)
    ax[2].set_yscale("log")
    # let the axis reach the quietest crossing instead of cutting it off
    _q = cr["QJ_mean"].replace(0, float("nan")).dropna()
    if len(_q):
        ax[2].set_ylim(float(_q.min()) * 0.5, float(_q.max()) * 3.0)

    if g is not None:
        med, mat = g
        for i, st in enumerate(mat.columns):
            ax[3].plot(mat.index, mat[st], color="gray", lw=0.5, alpha=0.5)
        ax[3].plot([], [], color="gray", lw=0.5, alpha=0.5,
                   label="individual stations")
        ax[3].plot(med.index, med, color="#d62728", lw=1.6,
                   label="median over %d stations" % mat.shape[1])
        ax[3].legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax[3].set_ylabel(a.metric)
    ax[3].set_xlabel("time (UT)")
    ax[3].margins(y=0.10)   # keep the per-station traces clear of the frame

    for x in ax:
        x.grid(alpha=0.25, lw=0.4)
        x.axvspan(pd.Timestamp("2015-03-17 04:00"), pd.Timestamp("2015-03-17 20:00"),
                  color="orange", alpha=0.10, lw=0)
    # In-figure title and explanatory note removed; the caption describes the event.
    fig.subplots_adjust(hspace=0.12, top=0.98, bottom=0.105)
    png = out / ("case_2015_storm_%s.png" % a.metric)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("[cs] figure -> %s" % png)

    # --- correlation between Q_J(t) and ground entropy (crossing epochs) ---
    if g is not None and not cr.empty:
        med = g[0].rename(a.metric)
        cr2 = cr.set_index("t_mid")
        j = pd.merge_asof(cr2.sort_index(), med.sort_index(),
                          left_index=True, right_index=True, direction="nearest",
                          tolerance=pd.Timedelta("30min"))
        j = j.dropna(subset=["QJ_mean", a.metric])
        if len(j) > 8:
            r = j["QJ_mean"].corr(j[a.metric], method="spearman")
            print("[cs] Spearman(Q_J per crossing, %s) = %+.3f  (n=%d)"
                  % (a.metric, r, len(j)))
        j.to_csv(out / ("case_2015_joint_%s.csv" % a.metric), index=False)
        print("[cs] joint table -> case_2015_joint_%s.csv" % a.metric)


if __name__ == "__main__":
    main()
