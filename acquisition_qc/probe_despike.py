#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_despike.py -- recover the despiking rule used by the processed ZW product.

verify_zwraw.py showed that E_Pc5 agrees exactly (corr = 1.000000) while the
entropy measures do not (CH_C corr = 0.44).  That is consistent with the processed
product having removed spikes before anything else: a band-pass energy at
1.6-6.7 mHz is insensitive to isolated single-sample spikes, whereas every entropy
measure in the battery is computed on the raw window and is extremely sensitive.

This probe lines the two H series up sample by sample and characterises the
differences, so the same rule can be applied to the 2023-2024 raw files -- which
is required, because mixing a despiked decade with an undes piked one would make
the years incomparable.
"""
from __future__ import annotations

import glob
from datetime import datetime

import numpy as np
import pandas as pd

RAW = "/data/ssd-stor/fsf/workspaces/project_yuliaoku/products/YLK/" \
      "YLK-BUFFER/NSSC-FSF/B015/dataset/data/zw1_fgm_dat"
PROC = "/data/ssd-stor/fsf/databases/data_ulf/dataset_pc/ZW"


def raw_day(station, day, nhead=9):
    f = "%s/%s/%d/%s_FGM01_DMD_L01_01D_%s000000.dat" % (
        RAW, station, day.year, station, day.strftime("%Y%m%d"))
    toks = open(f, errors="ignore").read().split()
    body = toks[nhead:]
    n = len(body) // 4
    h = np.full(n, np.nan)
    for k in range(n):
        v = body[4 * k]
        if v != "NULL":
            try:
                h[k] = float(v)
            except ValueError:
                pass
    return pd.Series(h, index=pd.date_range(day, periods=n, freq="s"))


def proc_day(station, day):
    """The processed product stores 15-min chunks; re-assemble one day."""
    pats = ["%s/%s/%d/*/feature_1Hz/**/*.csv" % (PROC, station, day.year)]
    files = []
    for p in pats:
        files += glob.glob(p, recursive=True)
    keep = []
    for f in files:
        try:
            t = pd.to_datetime(f.split("/")[-1].replace(".csv", ""))
        except Exception:
            continue
        if t.date() == day.date():
            keep.append((t, f))
    if not keep:
        return None
    ss = []
    for t, f in sorted(keep):
        d = pd.read_csv(f)
        hc = next((c for c in d.columns
                   if c.upper() in ("H", "DH")
                   or ("H" in c.upper() and "TIME" not in c.upper())), None)
        if hc is None:
            return None
        v = pd.to_numeric(d[hc], errors="coerce").to_numpy()
        ss.append(pd.Series(v, index=pd.date_range(t, periods=len(v), freq="s")))
    s = pd.concat(ss)
    return s[~s.index.duplicated()]


def main():
    st, day = "MHT", datetime(2015, 6, 15)
    r = raw_day(st, day)
    p = proc_day(st, day)
    print("raw  samples=%d  range %.2f .. %.2f" % (r.size, np.nanmin(r), np.nanmax(r)))
    if p is None:
        print("no processed day found for %s; trying another day" % day.date())
        for d in range(10, 20):
            p = proc_day(st, datetime(2015, 6, d))
            if p is not None:
                day = datetime(2015, 6, d)
                r = raw_day(st, day)
                print("using %s: proc samples=%d" % (day.date(), p.size))
                break
    if p is None:
        print("cannot locate a processed day")
        return

    common = r.index.intersection(p.index)
    print("common samples: %d" % len(common))
    a = r.loc[common].to_numpy()
    b = p.loc[common].to_numpy()
    diff = b - a
    nz = np.abs(diff) > 1e-9
    print("\n=== sample-by-sample comparison ===")
    print("  identical           : %d  (%.4f %%)" % ((~nz).sum(), 100.0 * (~nz).mean()))
    print("  differing           : %d  (%.4f %%)" % (nz.sum(), 100.0 * nz.mean()))
    if nz.sum():
        dd = diff[nz]
        print("  |diff| median       : %.3f nT" % np.median(np.abs(dd)))
        print("  |diff| p90 / max    : %.3f / %.3f nT" % (np.percentile(np.abs(dd), 90),
                                                          np.abs(dd).max()))
        print("\n  --- first 10 differing samples ---")
        idx = np.where(nz)[0][:10]
        for i in idx:
            print("    %s  raw=%9.3f  proc=%9.3f  diff=%+8.3f"
                  % (common[i].strftime("%H:%M:%S"), a[i], b[i], diff[i]))

    # is the processed series simply the raw series with single-sample outliers
    # replaced (e.g. by interpolation)?  Check whether differing samples are
    # isolated (neighbours identical) and whether proc lies between them.
    print("\n=== are the differing samples isolated? ===")
    iso = 0
    tot = 0
    for i in np.where(nz)[0]:
        if 1 <= i < len(a) - 1:
            tot += 1
            if not nz[i - 1] and not nz[i + 1]:
                iso += 1
    if tot:
        print("  isolated (both neighbours identical): %d / %d = %.1f %%"
              % (iso, tot, 100.0 * iso / tot))

    # spike detection on the raw series: how many samples exceed a local
    # time-gradient threshold?
    print("\n=== raw-series spike statistics (time-gradient criterion) ===")
    d1 = np.abs(np.diff(a))
    for thr in (5, 10, 20, 50, 100):
        print("  |dH/dt| > %-4d nT/s : %6d samples  (%.4f %%)"
              % (thr, (d1 > thr).sum(), 100.0 * (d1 > thr).mean()))


if __name__ == "__main__":
    main()
