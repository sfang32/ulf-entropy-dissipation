#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""omni_2015_ascii.py -- load 2015 OMNI 1-min indices from the fast OL-CB ASCII
mirror instead of the stalled /data/ftp CDF.

OMNI 1-min ASCII column positions (0-based, after whitespace split).
Verified against the identity AE = AU - AL on sample rows.
   37: AE_INDEX   38: AL   39: AU   40: SYM/D   41: SYM_H
Missing values are masked with |x| >= 9.9e4 (OMNI fill sentinel).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ASC = Path("/data/ssd-stor/OL-CB/DataCollection/AS-01-019/"
           "high_res_omni/omni_min2015.asc")

IDX_AE = 37
IDX_AL = 38
IDX_AU = 39
IDX_SYMD = 40
IDX_SYMH = 41


def _parse() -> pd.DataFrame:
    rows, tcol = [], []
    with open(ASC) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 46:
                continue
            try:
                year, doy, hr, mn = (int(p[0]), int(p[1]),
                                     int(p[2]), int(p[3]))
            except ValueError:
                continue
            ts = (pd.Timestamp(year, 1, 1)
                  + pd.Timedelta(days=doy - 1, hours=hr, minutes=mn))
            def _v(i):
                try:
                    x = float(p[i])
                except ValueError:
                    return np.nan
                return np.nan if abs(x) >= 9.9e4 else x
            tcol.append(ts)
            rows.append((_v(IDX_AE), _v(IDX_AL), _v(IDX_AU), _v(IDX_SYMH)))
    df = pd.DataFrame(rows, columns=["AE_INDEX", "AL", "AU", "SYM_H"],
                     index=pd.DatetimeIndex(tcol))
    df.index.name = "time"
    return df


def load_ae_10min() -> pd.DataFrame:
    """Return a DataFrame with columns ['time','ae'] at 10-min resolution
    (mirrors the original load_ae() resample)."""
    df = _parse()
    a = df["AE_INDEX"].resample("10min").mean().to_frame(name="ae")
    a = a.reset_index()          # 'time' becomes a plain column; index is nameless
    return a


def load_omni_range(t0, t1) -> pd.DataFrame:
    """Return 1-min OMNI for [t0, t1] with columns ['time','SYM_H','AE_INDEX']
    (mirrors the original CDF load_omni() interface)."""
    df = _parse()
    lo, hi = pd.Timestamp(t0), pd.Timestamp(t1)
    sub = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
    sub = sub.reset_index()       # 'time' becomes a plain column; index is nameless
    return sub[["time", "SYM_H", "AE_INDEX"]]


if __name__ == "__main__":
    df = _parse()
    mar = df[(df.index >= "2015-03-01") & (df.index < "2015-04-01")]
    print("rows total:", len(df))
    print("March 2015 SYM_H min/max: %.1f / %.1f" % (mar["SYM_H"].min(),
                                                      mar["SYM_H"].max()))
    print("March 2015 AE  min/max: %.1f / %.1f" % (mar["AE_INDEX"].min(),
                                                    mar["AE_INDEX"].max()))
    chk = df.dropna(subset=["AE_INDEX", "AL", "AU"])
    ok = np.allclose(chk["AE_INDEX"], chk["AU"] - chk["AL"], atol=1.0)
    print("AE == AU-AL within 1 nT:", ok)
