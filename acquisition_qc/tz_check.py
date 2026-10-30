#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tz_check.py -- is the ground (Meridian ZW) time axis UTC or Beijing time?

Critical: Swarm / OMNI are UTC. If the ground 10-min stamps are UTC+8, every
ground-space match is off by 8 h. Test = cross-correlation of the ground
entropy series against OMNI AE/AE over +-12 h of lag; the peak tells us the offset.

Run (server):  python tz_check.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import cdflib

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
OMNI = Path("/data/ftp/fsf/databases/data_omni/omni_1min/2015")
ST = ["MHT", "MZL", "NAT", "SSL", "MLS", "JFT", "LAT", "PXT", "SYS"]


def ground_series(metric="WPE"):
    fr = []
    for st in ST:
        f = RES / ("mb_zw_%s2015_w600_multiband.csv" % st)
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=lambda c: c in ("time", metric))
        d["time"] = pd.to_datetime(d["time"])
        fr.append(d.set_index("time")[metric].rename(st))
    m = pd.concat(fr, axis=1)
    return m.median(axis=1)


def omni_series():
    fr = []
    for f in sorted(glob.glob(str(OMNI / "omni_hro_1min_2015*.cdf"))):
        c = cdflib.CDF(f)
        t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
        out = {}
        for v, n in (("AE_INDEX", "ae"), ("SYM_H", "symh")):
            x = np.asarray(c.varget(v), dtype=float)
            out[n] = np.where(np.abs(x) > 1e30, np.nan, x)
        fr.append(pd.DataFrame(out, index=t))
    o = pd.concat(fr).sort_index()
    return o[~o.index.duplicated()]


def main():
    g = ground_series("WPE").resample("10min").mean()
    o = omni_series().resample("10min").mean()
    print("[tz] ground %d pts  %s .. %s" % (len(g), g.index[0], g.index[-1]))
    print("[tz] omni   %d pts  %s .. %s" % (len(o), o.index[0], o.index[-1]))
    print("\nlag[h]  rho(WPE,AE)   n     | rho(WPE,|SYM-H|)")
    best = None
    for lag in range(-12, 13):
        a = g.shift(-lag * 6)                       # +lag h trial offset
        j = pd.concat([a.rename("wpe"), o["ae"].rename("ae"),
                       o["symh"].abs().rename("asym")], axis=1).dropna()
        if len(j) < 1000:
            continue
        r1 = j.wpe.corr(j.ae, method="spearman")
        r2 = j.wpe.corr(j.asym, method="spearman")
        print("%+5d   %+.3f       %6d |  %+.3f" % (lag, r1, len(j), r2))
        if best is None or abs(r1) > abs(best[1]):
            best = (lag, r1)
    print("\n[tz] strongest |rho| at lag = %+d h  (rho=%+.3f)" % best)
    print("[tz] => ground clock is %s"
          % ("UTC" if best[0] == 0 else
             ("UTC+%d h (Beijing?)" % -best[0] if best[0] < 0
              else "UTC-%d h" % best[0])))


if __name__ == "__main__":
    main()
