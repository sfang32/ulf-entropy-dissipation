#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_zw_raw.py -- verify the B015 Meridian-I raw .dat layout against the
already-processed 1 Hz product, before using it for 2023-2024.

Background.  The analysis used `/data/ssd-stor/fsf/databases/data_ulf/dataset_pc/ZW`,
a processed 1 Hz "feature" product that stops at 2022.  The raw archive
(B015/dataset/data/zw1_fgm_dat) covers 2010-2024 in the SAME format, so the
question "why were 2023-2024 not analysed" has the answer "because the processed
product was never extended", not "because data are missing".

This probe does two things:
  1. works out the token layout of the daily file (how many header tokens, then
     a stream of 4-value groups);
  2. parses 2015 MHT with the inferred layout and compares it, sample by sample,
     against the processed 1 Hz CSV series used in the paper.  If the two agree,
     the raw reader is correct and can be trusted for 2023-2024.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

RAW = "/data/ssd-stor/fsf/workspaces/project_yuliaoku/products/YLK/" \
      "YLK-BUFFER/NSSC-FSF/B015/dataset/data/zw1_fgm_dat"
PROC = "/data/ssd-stor/fsf/databases/data_ulf/dataset_pc/ZW"


def probe_tokens(path):
    toks = open(path, errors="ignore").read().split()
    print("  %s" % os.path.basename(path))
    print("    tokens=%d" % len(toks))
    print("    head:", toks[:14])
    for h in range(6, 14):
        n = len(toks) - h
        if n > 0 and n % 4 == 0:
            print("    header=%2d -> %d groups = %d s  (86400 s in a day)"
                  % (h, n // 4, n // 4))
    return toks


def load_raw_day(path, nhead, scale=1.0):
    """One daily file -> (H series in nT, timestamps) using the given header size."""
    toks = open(path, errors="ignore").read().split()
    date = toks[0]
    body = toks[nhead:]
    n = len(body) // 4
    arr = np.full((n, 4), np.nan)
    for k in range(n):
        for c in range(4):
            v = body[4 * k + c]
            if v != "NULL":
                try:
                    arr[k, c] = float(v)
                except ValueError:
                    pass
    idx = pd.date_range(pd.to_datetime(date), periods=n, freq="s")
    return pd.Series(arr[:, 0] * scale, index=idx), arr


def main():
    print("=== (1) token layout of the raw daily file ===")
    f23 = sorted(glob.glob("%s/MHT/2023/*.dat" % RAW))[0]
    f15 = sorted(glob.glob("%s/MHT/2015/*.dat" % RAW))[0]
    probe_tokens(f15)
    probe_tokens(f23)

    print("\n=== (2) is the raw 2015 series the same thing as the processed product? ===")
    # processed 1 Hz feature csv used by the paper
    base = "%s/MHT/2015/10min/feature_1Hz" % PROC
    csvs = sorted(glob.glob(base + "/*.csv"))
    print("  processed csv files: %d" % len(csvs))
    if not csvs:
        print("  no processed files -- cannot cross-check")
        return
    d0 = pd.read_csv(csvs[0])
    print("  processed columns: %s" % list(d0.columns))
    print("  processed rows in first file: %d" % len(d0))
    print("  processed head:")
    print(d0.head(3).to_string(index=False))

    # raw for the same day
    day = os.path.basename(csvs[0]).replace(".csv", "")
    rf = "%s/MHT/2015/%s.dat" % (RAW, day.replace("-", ""))
    if not os.path.exists(rf):
        cands = [p for p in glob.glob("%s/MHT/2015/*.dat" % RAW)
                 if os.path.basename(p).split("_")[-1][:8] == day.replace("-", "")]
        rf = cands[0] if cands else None
    if not rf:
        print("  no raw file for %s" % day)
        return
    print("\n  raw file: %s" % os.path.basename(rf))
    ser, arr = load_raw_day(rf, 9)
    print("  raw parsed: %d samples, H range %.2f .. %.2f (nT)"
          % (ser.size, np.nanmin(ser.values), np.nanmax(ser.values)))

    print("\n  --- first 6 values side by side ---")
    hc = next((c for c in d0.columns if c.upper() in ("H", "DH")), d0.columns[1])
    print("   raw H : %s" % np.array2string(ser.values[:6], precision=2))
    print("   proc %s: %s" % (hc, np.array2string(
        pd.to_numeric(d0[hc], errors="coerce").to_numpy()[:6], precision=2)))

    # the processed series may be the raw series minus a daily baseline
    p = pd.to_numeric(d0[hc], errors="coerce").to_numpy()
    n = min(len(p), ser.size)
    r = ser.values[:n]
    print("\n  --- statistics over %d common samples ---" % n)
    print("   corr(raw, proc)          = %.8f" % np.corrcoef(r[~np.isnan(r)], p[~np.isnan(r)])[0, 1])
    d = p - r
    print("   mean(proc - raw)         = %+.3f   std = %.3f" % (np.nanmean(d), np.nanstd(d)))
    dr = p - (r - np.nanmean(r) + np.nanmean(p))
    print("   after removing the offset: std = %.3f  (0 => raw is the processed series" % np.nanstd(dr))
    print("                                                   up to a constant)")


if __name__ == "__main__":
    main()
