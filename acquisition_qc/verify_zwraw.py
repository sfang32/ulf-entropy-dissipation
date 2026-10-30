#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_zwraw.py -- does the raw-file reader reproduce the processed-product result?

The paper's 2011-2022 numbers come from the processed ZW tree; 2023-2024 must come
from the raw daily files.  Before mixing the two, confirm that reading MHT 2015
from the raw files gives the same answer as the processed tree did.  If it does,
the two sources are interchangeable and the new years can be appended.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RES = "/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/" \
      "v8_entropy_dissipation/results"

PAIRS = [("mb_zw_MHT2015_w600_multiband.csv", "processed (paper)"),
         ("mb_zwraw_MHT2015_w600_multiband.csv", "raw B015 files")]

METRICS = ["WPE", "SampEn", "SE_full", "CH_C", "Tsallis_q1.5"]


def main():
    frames = {}
    for fn, label in PAIRS:
        d = pd.read_csv("%s/%s" % (RES, fn))
        d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
        d = d.set_index("time").sort_index()
        frames[label] = d
        print("%-18s rows=%d  %s .. %s" % (label, len(d),
              d.index[0].strftime("%Y-%m-%d"), d.index[-1].strftime("%Y-%m-%d")))

    a, b = frames["processed (paper)"], frames["raw B015 files"]
    common = a.index.intersection(b.index)
    print("\ncommon windows: %d" % len(common))
    if len(common) < 1000:
        print("too few common windows")
        return

    print("\n=== agreement of the per-window quantities ===")
    hdr = "%-14s %10s %10s %10s" % ("quantity", "corr", "mean(A)", "mean(B)")
    print(hdr)
    for col in ["E_Pc5", "E_all", "dB_rms", "Q_all"] + METRICS:
        if col not in a.columns or col not in b.columns:
            continue
        x = pd.to_numeric(a.loc[common, col], errors="coerce").to_numpy()
        y = pd.to_numeric(b.loc[common, col], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            continue
        r = np.corrcoef(x[m], y[m])[0, 1]
        print("%-14s %10.6f %10.4f %10.4f  (n=%d)"
              % (col, r, x[m].mean(), y[m].mean(), int(m.sum())))

    print("\n=== the quantity the paper reports: rho(metric, log E_Pc5) ===")
    la = np.log10(pd.to_numeric(a.loc[common, "E_Pc5"], errors="coerce").clip(1e-12))
    lb = np.log10(pd.to_numeric(b.loc[common, "E_Pc5"], errors="coerce").clip(1e-12))
    print("%-14s %10s %10s %8s" % ("metric", "rho(A)", "rho(B)", "delta"))
    for k in METRICS:
        if k not in a.columns or k not in b.columns:
            continue
        va = pd.to_numeric(a.loc[common, k], errors="coerce")
        vb = pd.to_numeric(b.loc[common, k], errors="coerce")
        ra = va.corr(la, method="spearman")
        rb = vb.corr(lb, method="spearman")
        flag = "" if abs(ra - rb) < 0.02 else "  <-- DIFFERS"
        print("%-14s %+10.3f %+10.3f %8.3f%s" % (k, ra, rb, rb - ra, flag))

    print("\n=== verdict ===")
    rE = np.corrcoef(pd.to_numeric(a.loc[common, "E_Pc5"], errors="coerce")
                     .fillna(0).to_numpy(),
                     pd.to_numeric(b.loc[common, "E_Pc5"], errors="coerce")
                     .fillna(0).to_numpy())[0, 1]
    print("corr(E_Pc5) = %.6f  ->  %s"
          % (rE, "interchangeable" if rE > 0.99 else
             "NOT interchangeable -- investigate before mixing years"))


if __name__ == "__main__":
    main()
