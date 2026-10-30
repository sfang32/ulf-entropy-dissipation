"""diag_e4.py -- why did the 106-file epoch differ from the nine-file epoch?

`e4_storm.py` reindexes every station series onto the epoch grid of the 2015
storms, so a file from another year should contribute an all-NaN row that
`np.nanmedian(axis=0)` then simply ignores.  Yet `logs/rebuild_e4c.log`
(106 files, loose glob) reports a main-phase SampEn rise of +5.5 % whereas
`logs/e4_run.log` and `logs/e4_pinned.log` (nine files) report +22.7 %.

This script reuses e4_storm's own loaders so the epoch definition cannot drift,
reproduces both stacks, and prints per-column pre/main plus the number of rows
that actually carry data -- the quantity that decides whether the extra rows are
inert.  Run it from the code/ directory with the anaconda python.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

import e4_storm as e4

RES = e4.RES
COLS = ["E_Pc5", "E_all", "SE_full", "WPE", "SampEn", "Tsallis_q1.5"]


def stack(files, events, offs):
    curves = {c: [] for c in COLS}
    for f in files:
        df = pd.read_csv(f)
        if "E_Pc5" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None) \
            .dt.floor("10min")
        s = df.set_index("time")
        bad = pd.Series(False, index=s.index)
        for c in ("E_Pc5", "E_all"):
            if c in s.columns:
                bad |= (s[c] > 1e4)
        s.loc[bad, [c for c in COLS if c in s.columns]] = np.nan
        for c in COLS:
            if c not in s.columns:
                continue
            acc = []
            for t0 in events:
                seg = s[c].reindex(pd.date_range(t0, periods=len(offs),
                                                 freq="10min"))
                acc.append(seg.to_numpy())
            curves[c].append(np.nanmean(np.vstack(acc), axis=0))
    return curves


def summary(tag, curves, h):
    mpre = (h < 0)
    mmain = (h >= 0) & (h <= 6)
    a = np.vstack(curves["E_Pc5"])
    ok = np.isfinite(a).sum(axis=0)
    nz = int((np.isfinite(a).sum(axis=1) > 0).sum())
    print("=== %s : %d rows x %d bins ; rows carrying data = %d"
          % (tag, a.shape[0], a.shape[1], nz))
    print("    median contributing rows per bin: pre %d, main %d"
          % (int(np.median(ok[mpre])), int(np.median(ok[mmain]))))
    for c in COLS:
        v = np.vstack(curves[c])
        m = np.nanmedian(v, axis=0)
        p, q = np.nanmean(m[mpre]), np.nanmean(m[mmain])
        print("    %-12s pre=%.4f  main=%.4f  change=%+.2f%%"
              % (c, p, q, 100.0 * (q - p) / p))
    print()


def main():
    omni = e4.load_omni()
    sym = omni.set_index("time")["symh"]
    events = e4.find_storms(sym)
    offs = np.arange(-e4.PRE_H * 6, e4.POST_H * 6 + 1)
    h = offs / 6.0
    print("storms = %d   (thresh %.0f, gap %d h)\n" % (len(events), e4.THRESH, e4.GAP_H))

    nine = sorted(glob.glob(str(RES / "mb_zw_*2015_w600_multiband.csv")))
    allf = sorted(glob.glob(str(RES / "mb_zw_*_w600_multiband.csv")))
    print("nine-file set = %d ; loose glob = %d\n" % (len(nine), len(allf)))

    summary("nine 2015 files", stack(nine, events, offs), h)
    summary("loose glob (all years)", stack(allf, events, offs), h)


if __name__ == "__main__":
    main()
