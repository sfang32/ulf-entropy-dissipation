#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e2_sigma_multiyear.py -- extend the Fig.13 coupling matrix beyond the 2015
discovery year.

Motivation (handoff 5.2b): the manuscript is faulted for showing the core
band-coupling analysis on a single year.  The per-year Gate-3 files already
exist for every Phase-I year and both Phase-II years, so the fix is a
summation, not a re-run:

  results/e2_sigma_zw{2011..2022}.csv      Phase-I, 9 stations (2021-22: 8)
  results/e2_sigma_zwraw{2023,2024}.csv    Phase-I, raw zw ingest
  results/e2_sigma_zw22025.csv             Phase-II, 28 stations
  results/e2_sigma_zw22026.csv             Phase-II, 28 stations

Only the AE-modulated conductance rows are used (`scheme == "AE"`).  The three
schemes in these files are `const` (fixed 5 S), `AE` (5.0*sqrt(AE/100)) and `PC`
(5.0*sqrt(PC/1.0)); they share the same entropy inputs and differ only in the
Pedersen conductance prescription.  `const` is degenerate by construction -- it
makes P a constant multiple of the energy, so p_P|E is identically zero -- and
is kept in the file only as the implementation check reported in Sec. 3.2.
The paper's Gate-3 number is the `p_P_given_E` column of the `AE` scheme.

Note on scope: these per-year files carry only the Pc5 and broadband (`all`)
bands, not Pc3/Pc4 -- the multi-year panel is therefore Pc5 + broadband.

Outputs:
  results/e2_multiyear_matrix.csv   metric x band x year, station-mean p_P|E
  results/e2_multiyear_summary.csv  per metric: mean/min/max/sd/n_years, sign stability
  results/e2_multiyear_years.png    sign-stability panel: p_P|E vs year, one line per band
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
            "v8_entropy_dissipation/results")

# station count per data set, used only for the n column in the report
NST = {2011: 9, 2012: 9, 2013: 9, 2014: 9, 2015: 9, 2016: 9, 2017: 9,
       2018: 9, 2019: 9, 2020: 9, 2021: 8, 2022: 8, 2023: 9, 2024: 9,
       2025: 28, 2026: 28}
PHASE2 = {2025, 2026}

# the six metrics the paper actually reports in the Fig.13 matrix
KEEP = ["SE_full", "SE_Pc3", "SE_Pc4", "SE_Pc5", "PE", "WPE",
        "SampEn", "Tsallis_q1.5", "CH_C", "SVD", "IR_ordinal"]
# CH_H is byte-identical to PE (same order-type distribution embedded the same
# way), so it is dropped here to keep the multi-year panel readable.


def load_all(res: Path) -> pd.DataFrame:
    """Read every pinned per-year Gate-3 file -> tidy long frame."""
    frames = []
    files = sorted(res.glob("e2_sigma_zw*.csv"))
    if not files:
        raise SystemExit("[e2my] no e2_sigma_zw*.csv under %s" % res)
    for f in files:
        # Phase-II files are named e2_sigma_zw2{YYYY}.csv -- the leading "2" is a
        # generation tag, not part of the year, so a naive zw(\d{4}) reads 22025
        # as the year 2202.  Strip that tag first, then take the last 4 digits.
        m = re.search(r"zw(?:raw)?2?(\d{4})\.csv$", f.name)
        if not m:
            print("[e2my] %-28s SKIP (unrecognised name)" % f.name)
            continue
        year = int(m.group(1))
        if not (2010 <= year <= 2030):
            print("[e2my] %-28s SKIP (implausible year %d)" % (f.name, year))
            continue
        d = pd.read_csv(f)
        if "scheme" not in d.columns:
            continue
        d = d[d["scheme"].astype(str) == "AE"]
        if d.empty:
            print("[e2my] %-28s year=%d SKIP (no AE-scheme rows)" % (f.name, year))
            continue
        d = d[d["metric"].isin(KEEP)]
        # station-mean of p_P_given_E over the stations present in that year
        g = (d.dropna(subset=["p_P_given_E"])
               .groupby(["band", "metric"], as_index=False)
               .agg(rho_mean=("p_P_given_E", "mean"),
                    rho_min=("p_P_given_E", "min"),
                    rho_max=("p_P_given_E", "max"),
                    n_stations=("p_P_given_E", "size"),
                    n_win=("n", "sum")))
        g["year"] = year
        g["phase"] = "II" if year in PHASE2 else "I"
        frames.append(g)
        print("[e2my] %-28s year=%d rows=%d stations=%d"
              % (f.name, year, len(g), int(g["n_stations"].max())))
    if not frames:
        raise SystemExit("[e2my] no rows with scheme=AE")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["metric", "band", "year"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    res = Path(a.res)
    out = Path(a.outdir) if a.outdir else res
    out.mkdir(parents=True, exist_ok=True)

    t = load_all(res)
    t.to_csv(out / "e2_multiyear_matrix.csv", index=False)
    print("\n[e2my] matrix rows = %d, years = %s"
          % (len(t), sorted(t.year.unique())))

    # ---- per metric/band summary: sign stability across years -------------
    rows = []
    for (mt, bd), g in t.groupby(["metric", "band"]):
        v = g["rho_mean"].to_numpy(float)
        pos = int((v > 0).sum())
        rows.append({
            "metric": mt, "band": bd, "n_years": len(v),
            "mean": round(float(np.nanmean(v)), 4),
            "sd": round(float(np.nanstd(v, ddof=1)), 4) if len(v) > 1 else np.nan,
            "min": round(float(np.nanmin(v)), 4),
            "max": round(float(np.nanmax(v)), 4),
            "n_pos": pos, "n_neg": int((v < 0).sum()),
            "sign_stable": "yes" if pos in (0, len(v)) else "NO",
            "always_gt_0.2": "yes" if (v > 0.2).all() else "no",
            "always_lt_-0.2": "yes" if (v < -0.2).all() else "no",
        })
    s = pd.DataFrame(rows).sort_values(["band", "mean"],
                                       ascending=[True, False])
    s.to_csv(out / "e2_multiyear_summary.csv", index=False)
    print("\n=== sign stability across %d data sets ===" % t.year.nunique())
    print(s.to_string(index=False))

    # ---- the headline check: do SampEn / SE_full keep p_P|E >= 0.2? -------
    print("\n[e2my] broadband + Pc5 behaviour of the two surviving metrics")
    for mt in ("SampEn", "SE_full"):
        for bd in ("Pc5", "all"):
            g = t[(t.metric == mt) & (t.band == bd)]
            if g.empty:
                continue
            v = g.sort_values("year")["rho_mean"].to_numpy(float)
            print("  %-8s %-4s  mean=%+.3f  range=[%+.3f, %+.3f]  "
                  "n_pos=%d/%d  all>0.2=%s"
                  % (mt, bd, np.nanmean(v), np.nanmin(v), np.nanmax(v),
                     int((v > 0).sum()), len(v), bool((v > 0.2).all())))

    # ---- figure: sign stability vs year ---------------------------------
    bands = [b for b in ("Pc5", "all") if b in set(t.band)]
    mets = [m for m in KEEP if m in set(t.metric)]
    cmap = {"SampEn": "#1f77b4", "SE_full": "#d62728", "WPE": "#2ca02c",
            "Tsallis_q1.5": "#9467bd", "CH_C": "#ff7f0e", "SE_Pc5": "#8c564b"}
    fig, axes = plt.subplots(1, len(bands), figsize=(3.5 * len(bands), 3.9),
                             sharey=True)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12.5,
                         "axes.titlesize": 13, "xtick.labelsize": 10.5,
                         "ytick.labelsize": 10.5, "legend.fontsize": 10.5})
    years = sorted(t.year.unique())
    for k, bd in enumerate(bands):
        ax = axes[k]
        for mt in mets:
            g = t[(t.metric == mt) & (t.band == bd)]
            if g.empty:
                continue
            g = g.set_index("year").reindex(years)
            c = cmap.get(mt)
            if c is None:                     # grey out the supporting battery
                c, ls, lw, z = "#9aa0a6", ":", 1.0, 1
            else:
                ls, lw, z = "-", 2.0, 3
            ax.plot(years, g["rho_mean"].to_numpy(float), color=c, ls=ls,
                    lw=lw, alpha=1.0 if c != "#9aa0a6" else 0.75,
                    marker="o" if c != "#9aa0a6" else None, ms=3.6,
                    zorder=z, label=mt)
        ax.axhline(0.2, color="0.35", lw=0.9, ls="--", zorder=0)
        ax.axhline(0.0, color="0.55", lw=0.7, ls=":", zorder=0)
        ax.axhspan(-0.1, 0.1, color="0.85", alpha=0.55, zorder=-1)
        ax.set_title("band = %s" % ("broadband" if bd == "all" else bd))
        ax.set_xticks(years[::2])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.22, ls=":", lw=0.4)
    axes[0].set_ylabel(r"station-mean $p_{P\mid E}$")
    axes[-1].set_xlabel("data set (year)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=6,
               frameon=False, columnspacing=1.4, handletextpad=0.5)
    fig.subplots_adjust(wspace=0.10, top=0.86, bottom=0.24)
    fp = out / "e2_multiyear_years.png"
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[e2my] -> %s" % fp)
    print("[e2my] -> %s" % (out / "e2_multiyear_matrix.csv"))
    print("[e2my] -> %s" % (out / "e2_multiyear_summary.csv"))


if __name__ == "__main__":
    main()
