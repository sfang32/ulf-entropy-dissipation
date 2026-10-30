#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""station_consistency.py -- per-dataset across-station sign agreement (GATE 2).

For every metric x band, report how many stations share the majority sign of
Spearman(metric, log10 E_band).  This is the number quoted in the manuscript's
reproducibility-screening table, so it must be computed on the canonical
single-year / single-window files only (see the naming discipline note in
mb_summary.py).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

RES = "/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/" \
      "v8_entropy_dissipation/results"
DATASETS = [("D1 Phase-I 2015", "mb_zw_*2015_w600_multiband.csv", 9),
            ("D2 Phase-II 2025", "mb_zw2_*2025_w600_multiband.csv", 28),
            ("D3 Phase-II 2026", "mb_zw2_*2026_w600_multiband.csv", 28)]
METRICS = ["WPE", "SampEn", "SE_full", "CH_C", "PE", "CH_H",
           "Tsallis_q1.5", "IR_ordinal", "SVD"]
BANDS = ["Pc3", "Pc4", "Pc5", "all"]


def rhos(pattern, metric, band):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        col = "E_" + band
        if col not in d.columns or metric not in d.columns:
            continue
        ok = d[col].notna() & d[metric].notna()
        if ok.sum() < 100:
            continue
        r = d.loc[ok, metric].corr(np.log10(d.loc[ok, col].clip(1e-12)),
                                   method="spearman")
        if np.isfinite(r):
            out.append(r)
    return np.asarray(out)


def main():
    hdr = "%-14s %-5s" % ("metric", "band")
    for lab, _, n in DATASETS:
        hdr += " | %-22s" % ("%s (n=%d)" % (lab, n))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for m in METRICS:
        for b in BANDS:
            line = "%-14s %-5s" % (m, b)
            rec = {"metric": m, "band": b}
            for lab, pat, _ in DATASETS:
                r = rhos(pat, m, b)
                if r.size == 0:
                    line += " | %-22s" % "n/a"
                    continue
                neg = int((r < 0).sum())
                agree = max(neg, r.size - neg)
                sgn = "-" if neg * 2 > r.size else "+"
                line += " | %-22s" % ("%6.3f  %s %d/%d" % (r.mean(), sgn, agree, r.size))
                rec[lab] = (round(float(r.mean()), 4), agree, int(r.size))
            print(line)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "station_consistency.csv"), index=False)
    print("\n[save] station_consistency.csv")


if __name__ == "__main__":
    main()
