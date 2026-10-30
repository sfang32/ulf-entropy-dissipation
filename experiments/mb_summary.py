"""mb_summary.py -- cross-station multi-band summary (direction C core analysis).

Question it answers: WHICH band does each entropy metric actually track?
For every station x band x metric: Spearman(metric, log10 E_band), then aggregate
across stations and plot rho vs AACGM magnetic latitude (E5 prototype).

Outputs (results/):
  mb_summary.csv          long table: src, station, mlat, band, metric, rho, n
  mb_matrix.csv           pivot: metric x band (mean rho over stations)
  mb_heatmap.png          metric x band heatmap (mean rho, +-std annotated)
  mb_corr_vs_mlat.png     rho vs AACGM lat, one panel per band, 3 key metrics
Run on the server (anaconda3):
  python mb_summary.py [--include-sm]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
MLAT = {"MHT": 48.36, "MZL": 44.54, "NAT": 38.44, "SSL": 35.04, "MLS": 29.09,
        "PXT": 25.47, "JFT": 24.85, "HZT": 24.35, "LAT": 23.66, "SYT": 21.41,
        "SYS": 13.16, "KIR": 64.80, "YKC": 68.84, "BRW": 70.31, "SOD": 64.11,
        "ABK": 65.39, "GIM": 65.44, "NAQ": 64.83, "FCC": 67.74,
        # Meridian Phase-II FGMM (zw2; AACGM ref 2025-07-01)
        "OMOHE": 48.70, "OYAKS": 45.44, "OHUTB": 40.66, "OXLHT": 39.17,
        "OKSHI": 35.66, "OJIYG": 35.60, "OCSSL": 35.43, "OHLAN": 34.33,
        "ODLZS": 34.31, "OQIMO": 34.05, "OTYJY": 32.91, "OTANC": 29.47,
        "OGAER": 28.07, "OPIXI": 26.01, "OYICH": 25.78, "OENSH": 25.26,
        "OBATA": 25.09, "OHZXH": 24.73, "OLSCG": 24.72, "OSHYA": 21.90,
        "OLJGC": 21.80, "ONANA": 19.34, "OZHQI": 17.40, "OQIZH": 13.20,
        "OLEDO": 12.62,
        # CARISMA 1 Hz (aacgmv2 2.7.1, ref 2015-07-01; computed from the geographic
        # coordinates in each FGM header -- data/carisma/stations_carisma_aacgm.csv)
        "ANNA": 52.19, "OSAK": 55.28, "POLS": 54.34, "THRF": 57.28,
        "VULC": 57.25, "GULL": 57.76, "WGRY": 57.41, "WEYB": 58.07,
        "PINA": 59.41, "MSTK": 60.21, "LGRR": 61.23, "ISLL": 63.03,
        "OXFO": 64.02, "MCMU": 63.75, "GILL": 65.44, "DAWS": 65.77,
        "FCHP": 65.82, "FSMI": 66.89, "FSIM": 66.94, "RABB": 66.36,
        "BACK": 66.74, "FCHU": 67.74, "NORM": 69.34, "RANK": 71.64,
        "TALO": 77.76}
BANDS = ["Pc3", "Pc4", "Pc5", "all"]
# CH_H is deliberately absent: it is the same order-type distribution as PE
# embedded identically, so its station x band rho values are bit-for-bit equal to
# PE's (verified over all 42 station-band pairs, max |diff| = 0).  Plotting both
# would print the same row twice and make the heat map look like a 12-metric
# battery when the paper counts 11 independent measures.  The manuscript states
# the collapse explicitly.
METRICS = ["SE_full", "SE_Pc3", "SE_Pc4", "SE_Pc5", "PE", "WPE", "SampEn",
           "Tsallis_q1.5", "CH_C", "SVD", "IR_ordinal"]
KEY3 = ["SE_full", "WPE", "IR_ordinal"]


def src_of(basename):
    """Source token of an mb_*_multiband.csv file name.  Longest match first:
    `mb_zw2_...` must not be captured by the `mb_zw` prefix, and the older
    `mb_sm` arm is kept for the record even though CARISMA now supersedes it."""
    if basename.startswith("mb_carisma"):
        return "carisma"
    if basename.startswith("mb_sm"):
        return "sm"
    if basename.startswith("mb_zw2"):
        return "zw2"
    return "zw"


def station_of(fn):
    """MHT from mb_zw_MHT2015_w600_multiband.csv / OMOHE from mb_zw2_OMOHE2025_...
    / RANK from mb_carisma_RANK2015_w600_multiband.csv
    -- year-agnostic and src-agnostic (zw | zw2 | sm | carisma).
    The src token is matched as `[a-z0-9]+` rather than `[a-z0-9]{2,3}`: the
    {2,3} bound was written when the only sources were zw / zw2 / sm, and it
    silently failed on `carisma` (7 chars), which made station_of() fall through
    to the split('_')[2] branch and mis-key every CARISMA row."""
    m = re.search(r"mb_[a-z0-9]+_([A-Za-z0-9]+?)(\d{4})_", os.path.basename(fn))
    return m.group(1) if m else os.path.basename(fn).split("_")[2]


def year_of(fn):
    m = re.search(r"mb_[a-z0-9]+_[A-Za-z0-9]+?(\d{4})_", os.path.basename(fn))
    return int(m.group(1)) if m else 0


def win_of(fn):
    """Window tag, e.g. 'w600'.  Legacy files without one return None."""
    m = re.search(r"_(w\d+)_multiband\.csv$", os.path.basename(fn))
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-sm", action="store_true",
                    help="also include SuperMAG (E5b) runs")
    ap.add_argument("--include-carisma", action="store_true",
                    help="also include CARISMA 1 Hz runs.  These are 1 Hz, so they "
                         "share the ground --win (w600) and are directly comparable "
                         "with the ZW chain; this is the current high-latitude arm.")
    ap.add_argument("--carisma-months", default="",
                    help="restrict the CARISMA arm to these 1-12 month numbers "
                         "(comma separated, e.g. '1,2,3').  Pc5 power is strongly "
                         "seasonal, and the CARISMA archive is NOT uniformly "
                         "covering: of the 25 sites only 12 span all twelve months "
                         "of 2015, while BACK covers Oct-Dec only and TALO has 8 "
                         "days in January.  Pooling those unfiltered lets a "
                         "winter-vs-summer difference masquerade as a latitude "
                         "trend.  Leave empty to use every CARISMA day.")
    ap.add_argument("--drop-stations", default="THRF,TALO",
                    help="CARISMA sites to exclude as instrument outliers.  Both "
                         "were found by a robust latitude-neighbour test on the "
                         "station-median SE_full (median of the stations above and "
                         "below in magnetic latitude, MAD-scaled z > 5): THRF at "
                         "57.28 deg returns 0.724 against neighbours VULC 0.221 "
                         "(57.25 deg) and WGRY 0.217 (57.41 deg) -- a 3.3x excess "
                         "at0.03 deg of latitude separation, which no latitudinal "
                         "effect can produce, and it begins abruptly on 2015-05-30 "
                         "and persists for 216 days with no recovery, the signature "
                         "of a sensor state change rather than geophysics.  TALO "
                         "(77.76 deg, polar cap) contributes 8 days from January "
                         "only.  Override with an empty string to keep them.")
    ap.add_argument("--include-zw2", action="store_true",
                    help="also include Meridian Phase-II FGMM (zw2) runs")
    ap.add_argument("--year", type=int, default=0,
                    help="restrict to one year (0 = all). Needed when the same "
                         "station was run for several years (e.g. zw2 2025+2026).")
    ap.add_argument("--win", default="w600",
                    help="window tag for the ground networks (zw / zw2). Default "
                         "w600 = 10-min at 1 Hz.")
    ap.add_argument("--win-sm", default="w3600",
                    help="window tag for the SuperMAG extension (sm, E5b): w3600 = "
                         "1-h at 1 min. Kept separate from --win because the "
                         "window-sensitivity runs of the GROUND network are also "
                         "tagged w3600 (60-min at 1 Hz) and must not be confused "
                         "with the 1-min SuperMAG windows.")
    a = ap.parse_args()
    carisma_months = ([int(x) for x in a.carisma_months.split(",") if x.strip()]
                      if a.carisma_months else [])
    drop_stations = {s.strip().upper() for s in a.drop_stations.split(",") if s.strip()}
    if carisma_months and any(m < 1 or m > 12 for m in carisma_months):
        raise SystemExit("[mb] --carisma-months must be month numbers 1-12")

    files = sorted(glob.glob(str(RES / "mb_*_multiband.csv")))
    rows = []
    n_skipped = 0
    for f in files:
        b = os.path.basename(f)
        # 🔴 integrity guard: only the canonical window runs may enter the summary.
        # results/ also holds legacy smoke files (`mb_zw_MHT2015_multiband.csv`) and
        # the window-sensitivity runs (`..._w1200/2400/3600_...`); without this
        # filter those stations get counted several times and bias every mean.
        # The window tag is checked PER SOURCE: the ground window-sensitivity runs
        # are also tagged w3600 (60-min at 1 Hz) but are a different experiment
        # from the 1-min SuperMAG windows, so sm is matched against --win-sm.
        # CARISMA is 1 Hz like the ground network, so it matches --win.
        src = src_of(b)
        if win_of(b) != (a.win_sm if src == "sm" else a.win):
            n_skipped += 1
            continue
        if src == "sm" and not a.include_sm:
            continue
        if src == "carisma" and not a.include_carisma:
            continue
        if src == "zw2" and not a.include_zw2:
            continue
        if a.year:
            yy = year_of(b)
            if yy != a.year:
                continue
        st = station_of(f)
        if src == "carisma" and st in drop_stations:
            continue
        df = pd.read_csv(f)
        if "E_Pc5" not in df.columns or "WPE" not in df.columns:
            continue
        if src == "carisma" and a.carisma_months and "day" in df.columns:
            # `day` is an ISO date string written by run_multiband.py
            keep = df["day"].astype(str).str.slice(5, 7).astype(int) \
                        .isin(carisma_months)
            if keep.sum() < 1000:      # a station with <1000 windows in the
                continue                # requested months carries no weight
            df = df[keep]
        ok = df["E_Pc5"].notna() & df["WPE"].notna()
        s = df[ok]
        for b in BANDS:
            col = f"E_{b}"
            if col not in s.columns:
                continue
            m2 = s[col].notna()
            if m2.sum() < 100:
                continue
            le = np.log10(s.loc[m2, col].clip(1e-12))
            for k in METRICS:
                if k not in s.columns:
                    continue
                v = s.loc[m2, k]
                if v.notna().sum() < 100:
                    continue
                rows.append({"src": src, "station": st, "mlat": MLAT.get(st, np.nan),
                             "year": year_of(os.path.basename(f)),
                             "band": b, "metric": k,
                             "rho": round(v.corr(le, method="spearman"), 4),
                             "n": int(m2.sum())})
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit("no mb results found")
    print("[mb] win=%s  files scanned=%d  skipped=%d  station-rows=%d"
          % (a.win, len(files), n_skipped, len(tab)))
    # a station-year must contribute exactly ONE file (guard against duplicate
    # runs at different windows or legacy un-tagged outputs silently double-counting)
    seen = set()
    for f in files:
        b = os.path.basename(f)
        src2 = src_of(b)
        if src2 == "sm" and not a.include_sm:
            continue
        if src2 == "carisma" and not a.include_carisma:
            continue
        if src2 == "zw2" and not a.include_zw2:
            continue
        if win_of(b) != (a.win_sm if src2 == "sm" else a.win):
            continue
        if a.year and year_of(b) != a.year:
            continue
        key = (src2, station_of(b), year_of(b))
        if key in seen:
            raise SystemExit("[mb] duplicate run for %s -> %s" % (key, b))
        seen.add(key)
    print("[mb] win=%s  canonical runs=%d  (files scanned=%d, skipped=%d)"
          % (a.win, len(seen), len(files), len(files) - len(seen)))
    dup = tab.groupby(["src", "station", "year", "band", "metric"]).size()
    if (dup > 1).any():
        raise SystemExit("[mb] duplicate rows: %s" % dup[dup > 1].head().to_dict())
    tab.to_csv(RES / "mb_summary.csv", index=False)
    stations = sorted(tab["station"].unique())
    print("[save] mb_summary.csv  stations=%d (%s)" % (len(stations), " ".join(stations)))
    if carisma_months:
        ca = sorted(tab[tab["src"] == "carisma"]["station"].unique())
        print("[mb] CARISMA month filter = %s  -> %d stations: %s"
              % (sorted(carisma_months), len(ca), " ".join(ca)))
    if drop_stations:
        ca = sorted(tab[tab["src"] == "carisma"]["station"].unique())
        gone = sorted(s for s in drop_stations
                      if s not in {r["station"] for r in
                                   tab[tab["src"] == "carisma"].to_dict("records")})
        print("[mb] CARISMA instrument outliers dropped = %s  (excluded from the %d"
              " stations kept)" % (" ".join(gone), len(ca)))
        # Per-station month coverage, so a reviewer can see the seasonal basis of
        # the latitude trend instead of having to trust it.
        cov = {}
        for f in sorted(glob.glob(str(RES / "mb_carisma_*_multiband.csv"))):
            stc = station_of(f)
            if ca and stc not in ca:
                continue
            d = pd.read_csv(f, usecols=["day"])
            mo = sorted(set(d["day"].astype(str).str.slice(5, 7)))
            cov[stc] = (d["day"].nunique(), len(mo))
        for stc, (nd, nm) in sorted(cov.items(), key=lambda kv: MLAT.get(kv[0], 0)):
            print("    %-5s mlat=%5.2f  %3d days  %2d/12 months%s"
                  % (stc, MLAT.get(stc, float("nan")), nd, nm,
                     "" if nm == 12 else "   <-- PARTIAL COVERAGE"))

    # focus table: Phase-I if present, else whatever source actually has data
    # (e.g. `--year 2026` -> only zw2 exists)
    zw = tab[tab["src"] == "zw"]
    if zw.empty:
        srcs = list(tab["src"].unique())
        print("[mb] no zw rows -> using src=%s for the summary blocks"
              % "/".join(srcs))
        zw = tab
    mat = zw.pivot_table(index="metric", columns="band", values="rho", aggfunc="mean")
    mat = mat.reindex(index=[m for m in METRICS if m in mat.index], columns=BANDS)
    mat.round(3).to_csv(RES / "mb_matrix.csv")
    if a.year:                        # keep per-year copies for reproducibility tables
        tag = "_".join(sorted(zw["src"].unique())) + "_%d" % a.year
        mat.round(3).to_csv(RES / ("mb_matrix_%s.csv" % tag))
        print("[save] mb_matrix_%s.csv" % tag)
    print("\n=== mean Spearman(metric, log10 E_band) over %s stations ==="
          % "+".join(zw["src"].unique()))
    print(mat.round(3).to_string())
    if mat.empty:
        raise SystemExit("[mb] empty matrix -- nothing to plot")

    # --- heatmap v2 (Qwen review: clip colour scale to data, add significance,
    #     group rows, drop the crowding sd line) ---
    from scipy import stats as sstats
    ORDER = [["SE_full", "SE_Pc3", "SE_Pc4", "SE_Pc5"],
             ["PE", "WPE", "SampEn"],
             ["Tsallis_q1.5", "CH_C", "SVD"],
             ["IR_ordinal"]]
    ordered = [m for grp in ORDER for m in grp if m in mat.index]
    mat = mat.reindex(index=ordered)

    def star(rs):
        if rs.size < 6:
            return ""
        try:
            p = sstats.wilcoxon(rs).pvalue       # do the 9 station rho values
        except Exception:
            return ""
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5})
    fig, ax = plt.subplots(figsize=(5.8, 7.0))
    vmax = float(np.nanmax(np.abs(mat.to_numpy(dtype=float)))) * 1.05
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i, m in enumerate(mat.index):
        for j, b in enumerate(mat.columns):
            v = mat.iloc[i, j]
            if not np.isfinite(v):
                continue
            rs = zw[(zw.metric == m) & (zw.band == b)]["rho"].dropna().to_numpy()
            ax.text(j, i, "%.2f" % v, ha="center", va="center",
                    fontsize=10.5, color="white" if abs(v) > 0.6 * vmax else "black")
    # group separators
    cut = 0
    for grp in ORDER:
        cut += len([m for m in grp if m in mat.index])
        if cut < len(mat.index):
            ax.axhline(cut - 0.5, color="k", lw=1.0)
    # cell borders
    for x in range(len(mat.columns) + 1):
        ax.axvline(x - 0.5, color="white", lw=0.8)
    for y in range(len(mat.index) + 1):
        ax.axhline(y - 0.5, color="white", lw=0.8)
    # (same-band cell highlighting removed: the black rectangles read as orphaned
    #  borders once the significance stars were dropped; the caption explains the
    #  sign pattern.)
    # Title removed; the manuscript caption describes the heat map.
    fig.colorbar(im, ax=ax, label=r"Spearman $\rho$", shrink=0.75,
                   ticks=np.round([-vmax, -0.4, -0.2, 0.0, 0.2, 0.4, vmax], 2))
    fig.tight_layout()
    fig.savefig(RES / "mb_heatmap.png", dpi=300)
    print("[save] mb_heatmap.png  (colour range +-%.2f)" % vmax)

    # --- rho vs AACGM latitude, one panel per band ---
    # (IR_ordinal was dropped after failing the across-station gate; SampEn replaces it)
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    KEYLAT = ["SE_full", "WPE", "SampEn"]
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.0 * len(BANDS), 4.0),
                             sharey=True)
    # Okabe-Ito (colour-blind safe)
    colors = dict(zip(KEYLAT, ["#0072B2", "#D55E00", "#009E73"]))
    for ax, b in zip(np.atleast_1d(axes), BANDS):
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axhspan(-0.10, 0.10, color="gray", alpha=0.13, lw=0)
        for k in KEYLAT:
            sub = zw[(zw.band == b) & (zw.metric == k)].dropna(subset=["mlat", "rho"])
            sub = sub.sort_values("mlat")
            if sub.empty:
                continue
            ax.plot(sub["mlat"], sub["rho"], "-o", ms=5, lw=1.5, color=colors[k],
                    label=k, alpha=0.9)
        ax.set_xlabel("AACGM magnetic latitude [deg]")
        ax.set_title("band: %s" % ("broadband" if b == "all" else b))
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_ylim(-0.85, 0.45)
    axes[0].set_ylabel(r"Spearman $\rho$ (metric vs $\log_{10}E_{band}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    # the grey |rho| < 0.10 band was described only in the sub-title text; give it
    # a legend entry so the reader can identify it without reading the caption
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.13, ec="none"))
    labels.append(r"$|\rho| < 0.10$")
    fig.legend(handles, labels, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), frameon=False)
    # In-figure title removed; described in the manuscript caption.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RES / "mb_corr_vs_mlat.png", dpi=300, bbox_inches="tight")
    print("[save] mb_corr_vs_mlat.png")

    # --- best band per metric ---
    print("\n=== best band per metric (mean |rho| over ZW stations) ===")
    for m in mat.index:
        r = mat.loc[m].abs()
        print("  %-12s best=%-4s |rho|=%.3f   (Pc3 %+.2f Pc4 %+.2f Pc5 %+.2f all %+.2f)"
              % (m, r.idxmax(), r.max(), mat.loc[m, "Pc3"], mat.loc[m, "Pc4"],
                 mat.loc[m, "Pc5"], mat.loc[m, "all"]))


if __name__ == "__main__":
    main()
