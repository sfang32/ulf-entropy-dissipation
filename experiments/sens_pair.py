"""sens_pair.py -- matched-station 1 Hz vs 1 min comparison at 60-minute windows.

Motivation
----------
The sampling-rate decomposition in the manuscript compares

    1 Hz, 60 min  ->  means over MHT, NAT, SYS
    1 min, 60 min ->  means over MHT, NAT, MLS

i.e. two *different* station sets (see sens_window.csv, fs_tag column).  Part of
what looked like a sampling effect could therefore be a station effect.  This
script recomputes both rows on one identical nine-station set, which is the only
configuration in which the comparison supports the claim it is used for.

Inputs (produced by run_multiband.py --outdir results/sens9):
    mb_zw_{ST}2015_w3600_multiband.csv   1 Hz, 3600 samples = 60 min
    mb_zw_{ST}2015_w60_multiband.csv     1 min (--decimate 60), 60 samples = 60 min

Output:
    results/sens_pair.csv   station, fs_tag, band, metric, n, rho

Run on the server with the anaconda python.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
SRC = RES / "sens9"
ST = ["MHT", "MZL", "NAT", "SSL", "MLS", "JFT", "LAT", "PXT", "SYS"]
METRICS = ["SE_full", "WPE", "SampEn", "Tsallis_q1.5"]
BANDS = ["Pc3", "Pc4", "Pc5", "all"]

# [tag, filename template, label]
CONFIGS = [
    ("1 Hz",  "mb_zw_%s2015_w3600_multiband.csv", "1 Hz, 60 min"),
    ("1 min", "mb_zw_%s2015_w60_multiband.csv",   "1 min, 60 min"),
]


def extract(f, st, fs_tag):
    df = pd.read_csv(f)
    ok = df["E_Pc5"].notna() & df["WPE"].notna()
    s = df[ok]
    if len(s) < 100:
        return []
    out = []
    for b in BANDS:
        col = "E_%s" % b
        if col not in s.columns:
            continue
        m2 = s[col].notna()
        le = np.log10(s.loc[m2, col].clip(1e-12))
        for k in METRICS:
            if k not in s.columns or s.loc[m2, k].notna().sum() < 100:
                continue
            out.append({"station": st, "fs_tag": fs_tag, "band": b,
                        "metric": k, "n": int(m2.sum()),
                        "rho": float(s.loc[m2, k].corr(le, method="spearman"))})
    return out


def main():
    rows = []
    for st in ST:
        for tag, tpl, _ in CONFIGS:
            f = SRC / (tpl % st)
            if not f.exists():
                print("  [warn] missing %s" % f.name, file=sys.stderr)
                continue
            rows += extract(f, st, tag)
    if not rows:
        raise SystemExit("no inputs found in %s" % SRC)
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "sens_pair.csv", index=False)

    print("=== matched station set: %d stations x 2 configurations ==="
          % tab.station.nunique())
    print()
    hdr = "%-16s %-12s %8s %6s" % ("configuration", "metric", "Pc5", "all")
    print(hdr)
    print("-" * len(hdr))
    for tag, _, label in CONFIGS:
        for k in METRICS:
            sub = tab[(tab.fs_tag == tag) & (tab.metric == k)]
            if sub.empty:
                continue
            pc5 = sub[sub.band == "Pc5"].rho.mean()
            allb = sub[sub.band == "all"].rho.mean()
            print("%-16s %-12s %+8.3f %+6.3f" % (label, k, pc5, allb))
        print()

    print("=== Pc5 sign-flip check (per station) ===")
    for k in METRICS:
        a = tab[(tab.fs_tag == "1 Hz") & (tab.metric == k) & (tab.band == "Pc5")] \
            .set_index("station").rho
        b = tab[(tab.fs_tag == "1 min") & (tab.metric == k) & (tab.band == "Pc5")] \
            .set_index("station").rho
        j = pd.concat([a.rename("hz"), b.rename("min")], axis=1).dropna()
        if j.empty:
            continue
        flip = int(((j.hz < 0) != (j["min"] < 0)).sum())
        print("  %-12s n=%d  mean 1Hz=%+.3f -> 1min=%+.3f   sign flips=%d/%d"
              % (k, len(j), j.hz.mean(), j["min"].mean(), flip, len(j)))
        for st in j.index:
            print("      %-5s %+7.3f -> %+7.3f  %s"
                  % (st, j.hz[st], j["min"][st],
                     "FLIP" if (j.hz[st] < 0) != (j["min"][st] < 0) else ""))


if __name__ == "__main__":
    main()
