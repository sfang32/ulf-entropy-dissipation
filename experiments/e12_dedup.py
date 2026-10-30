#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e12_dedup.py -- count deduplicated ground windows for the Swarm-ground match.

Replicates the merge_asof logic of e12_swarm_ground.py (nearest ground window
within 30 min) and reports how many DISTINCT 600 s ground windows received at
least one Swarm crossing -- the effective (non-independent) sample size the
manuscript must report instead of the raw crossing count.
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
ST = ["MHT", "MZL", "NAT", "SSL", "MLS", "JFT", "LAT", "PXT", "SYS"]
METRICS = ["SE_full", "WPE", "SampEn", "CH_C", "Tsallis_q1.5"]


def ground_table(year, metrics=METRICS):
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
    return pd.DataFrame(out).sort_index()


def report(year, tol=30.0):
    cs = sorted(glob.glob(str(RES / ("swarm_crossings_%d*.csv" % year))))[0]
    cr = pd.read_csv(cs)
    cr["t_mid"] = pd.to_datetime(cr["t_mid"])
    cr = cr.sort_values("t_mid")
    g = ground_table(year)
    gt = g.reset_index().rename(columns={g.index.name or "time": "gtime"})
    print("[%d] crossings=%d  ground-window grid=%d"
          % (year, len(cr), len(gt)))
    j = pd.merge_asof(cr, gt, left_on="t_mid", right_on="gtime",
                      direction="nearest",
                      tolerance=pd.Timedelta(minutes=tol))
    j = j.dropna(subset=["QJ_mean", "WPE"])
    matched = len(j)
    n_win = j["gtime"].nunique()
    per = j.groupby("gtime").size()
    print("      matched crossings = %d" % matched)
    print("      DISTINCT ground windows = %d" % n_win)
    print("      crossings / window = %.3f (max=%d, median=%.1f)"
          % (matched / n_win, per.max(), per.median()))
    print("      => effective sample is %.1f%% of the raw crossing count"
          % (100.0 * n_win / matched))


def gt_index_name(gt):
    return gt.index.name or "time"


if __name__ == "__main__":
    for y in (2015, 2014):
        report(y)
