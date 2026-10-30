"""e4_storm.py -- E4: superposed-epoch analysis of the entropy-dissipation coupling
across geomagnetic storms (SYM-H minima as epoch zero).

Events are detected from the OMNI 2015 1-min SYM-H record (threshold -50 nT, minima
separated by >24 h), then every quantity is stacked on a -6 h .. +48 h grid using the
10-min multiband windows of the 9 Meridian stations (station mean + between-station SEM).

Panels: SYM-H (event definition), AE, band energies E_Pc5/E_all, and the entropy
metrics SE_full / WPE / SampEn (the three that passed E1/E3).

Run on the server: python e4_storm.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cdflib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
OMNI = Path("/data/ftp/fsf/databases/data_omni/omni_1min/2015")
THRESH = -50.0
GAP_H = 24
PRE_H, POST_H = 6, 48


def _var(c, name):
    x = np.asarray(c.varget(name), dtype=float)
    return np.where(np.abs(x) > 1e30, np.nan, x)


def load_omni():
    frames = []
    for f in sorted(glob.glob(str(OMNI / "omni_hro_1min_2015*_v01.cdf"))):
        c = cdflib.CDF(f)
        t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
        frames.append(pd.DataFrame({"symh": _var(c, "SYM_H"),
                                    "ae": _var(c, "AE_INDEX")}, index=t))
    d = pd.concat(frames).sort_index()
    d = d[~d.index.duplicated()]
    agg = d.resample("10min").mean()
    agg["time"] = agg.index
    return agg


def find_storms(symh, thresh=THRESH, gap_h=GAP_H):
    v = symh.to_numpy()
    idx = np.where(np.isfinite(v) & (v < thresh))[0]
    if idx.size == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    cand = [symh.index[g[np.argmin(v[g])]] for g in groups]
    keep = []
    for t in cand:
        if not keep or (t - keep[-1]).total_seconds() > gap_h * 3600:
            keep.append(t)
    return keep


def main():
    omni = load_omni()
    sym = omni.set_index("time")["symh"]
    events = find_storms(sym)
    print("[E4] %d storm events (SYM-H < %.0f nT, >%d h apart)" %
          (len(events), THRESH, GAP_H), flush=True)
    for t in events:
        print("   %s  SYM-H_min=%.0f nT" % (t, sym.loc[t]), flush=True)
    if not events:
        raise SystemExit("no events")

    # Epoch window: PRE_H hours before the storm minimum to POST_H hours after.
    # The `offs` grid is labelled h = offs/6, i.e. it starts at -PRE_H, so the
    # reindexed window MUST start at t0 - PRE_H -- not at t0.  Starting at t0 put
    # the individual minimum (data index 0) at the label h = -PRE_H, i.e. the whole
    # figure and the pre/main-phase comparison below were displaced by PRE_H = 6 h:
    # what the CSV called "pre-storm" was in fact the main phase, and what it called
    # "main phase" was the early recovery.  Verified on the first event: data index 0
    # = 2015-01-03 01:40 (= t0, -53.6 nT) was drawn at h = -6, while the label h = 0
    # sat on 07:40 (-30.8 nT).
    offs = np.arange(-PRE_H * 6, POST_H * 6 + 1)
    t_start = lambda t0: pd.date_range(t0 - pd.Timedelta(hours=PRE_H),
                                       periods=len(offs), freq="10min")

    # ---- stack OMNI drivers ----
    def stack(series):
        acc = []
        for t0 in events:
            s = series.reindex(t_start(t0).tz_localize(series.index.tz))
            acc.append(s.to_numpy())
        agg_ = np.vstack(acc)
        return (np.nanmedian(agg_, axis=0),
                (np.nanpercentile(agg_, 84, axis=0)
                 - np.nanpercentile(agg_, 16, axis=0)) / 2.0)

    sym_m, sym_e = stack(sym)
    ae_m, ae_e = stack(omni.set_index("time")["ae"])

    # ---- stack station quantities ----
    # Canonical runs only (see mb_summary.py for the naming discipline).  Two
    # years are needed, not one: the last 2015 storms reach their SYM-H minimum in
    # late December, so the +48 h part of their epoch lies in January 2016 and only
    # the 2016 records fill those bins (measured: the 2016 files carry 257 of the
    # 325 epoch bins).  A bare `mb_zw_*_w600_multiband.csv` glob used to pull this
    # in by accident -- together with sixteen other years, which happened to be
    # inert because their timestamps cannot align with a 2015 epoch -- so the set
    # is now declared explicitly and its size asserted.
    files = []
    for y in (2015, 2016):
        files += sorted(glob.glob(str(RES / ("mb_zw_*%d_w600_multiband.csv" % y))))
    assert len(files) == 18, \
        "expected 9 stations x {2015,2016}, found %d" % len(files)
    print("[E4] epoch completed by two calendar years: %d station-years" % len(files))
    cols = ["E_Pc5", "E_all", "SE_full", "WPE", "SampEn", "Tsallis_q1.5"]
    st_curves = {c: [] for c in cols}
    for f in files:
        df = pd.read_csv(f)
        if "E_Pc5" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None) \
            .dt.floor("10min")
        s = df.set_index("time")
        # DATA QUALITY: a few windows carry saturation/spike contamination
        # (E_pc5 up to 1e7 nT^2 <=> dB_rms ~ 3e3 nT, unphysical). Drop the WHOLE
        # window for every quantity so the epoch average is not dominated by them.
        # (E1/E3/E5 use Spearman and are robust to these; only mean-based stacks are not.)
        bad = pd.Series(False, index=s.index)
        for c in ("E_Pc5", "E_all"):
            if c in s.columns:
                bad |= (s[c] > 1e4)
        nbad = int(bad.sum())
        s.loc[bad, [c for c in cols if c in s.columns]] = np.nan
        print("   %-6s dropped %4d contaminated windows (%.3f%%)"
              % (os.path.basename(f).split("_")[2], nbad,
                 100.0 * nbad / max(len(s), 1)), flush=True)
        for c in cols:
            if c not in s.columns:
                continue
            acc = []
            for t0 in events:
                seg = s[c].reindex(t_start(t0))
                acc.append(seg.to_numpy())
            st_curves[c].append(np.nanmean(np.vstack(acc), axis=0))
    print("[E4] stacked %d stations" % len(st_curves["E_Pc5"]), flush=True)

    def agg(c):
        a = np.vstack(st_curves[c])
        return (np.nanmedian(a, axis=0),
                (np.nanpercentile(a, 84, axis=0)
                 - np.nanpercentile(a, 16, axis=0)) / 2.0)

    h = offs / 6.0  # hours relative to SYM-H minimum
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    panels = [("SYM-H", sym_m, sym_e, "#1f77b4", "SYM-H [nT]"),
              ("AE", ae_m, ae_e, "#ff7f0e", "AE [nT]")]
    for ax, (nm, m, e, c, lab) in zip(axes.ravel(), panels):
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.plot(h, m, color=c, lw=1.6)
        ax.fill_between(h, m - e, m + e, color=c, alpha=0.25, lw=0)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    for ax, (c, lab, logy) in zip(axes.ravel()[2:],
                                  [("E_Pc5", r"$E_{\rm Pc5}$ [nT$^2$]", True),
                                   ("E_all", r"$E_{\rm all}$ [nT$^2$]", True)]):
        m, e = agg(c)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.plot(h, m, color="#2ca02c", lw=1.6)
        ax.fill_between(h, m - e, m + e, color="#2ca02c", alpha=0.25, lw=0)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    for ax, (c, lab) in zip(axes.ravel()[4:],
                            [("SE_full", r"$SE_{\rm full}$"), ("WPE", "WPE"),
                             ("SampEn", r"$S_{\rm SampEn}$"),
                             ("Tsallis_q1.5", r"$T_{q=1.5}$")]):
        m, e = agg(c)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.plot(h, m, color="#d62728", lw=1.6)
        ax.fill_between(h, m - e, m + e, color="#d62728", alpha=0.25, lw=0)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0, 0].set_ylim(-95, 8)
    for ax in axes[-1]:
        ax.set_xlabel("hours relative to SYM-H minimum")
    for ax in axes.ravel():
        ax.set_xticks(np.arange(-5, 50, 5))
        ax.set_xlim(-5, 48)
        # Storm main phase runs from onset to the SYM-H minimum, i.e. it lies to the
        # LEFT of h = 0.  Shading 0..+9 labelled the recovery as the main phase, which
        # the 6-hour window misalignment had previously hidden (the plotted "0" sat on
        # the real h = +6).  With the window fixed, the shaded span and its label are
        # now on the correct side of the minimum.
        ax.axvspan(-PRE_H, 0, color="orange", alpha=0.13, lw=0)
    # legend colours must match the actual panel colours (blue SYM-H, orange AE,
    # green energy, red entropy); the IQR band takes the colour of its own panel
    fig.legend(handles=[plt.Line2D([], [], color="#1f77b4", lw=1.8),
                        plt.Line2D([], [], color="#ff7f0e", lw=1.8),
                        plt.Line2D([], [], color="#2ca02c", lw=1.8),
                        plt.Line2D([], [], color="#d62728", lw=1.8),
                        plt.Rectangle((0, 0), 1, 1, fc="0.75", alpha=0.45, ec="none"),
                        plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.25, ec="none")],
               labels=["median: SYM-H", "median: AE",
                       r"median: energy ($E_{Pc5}$, $E_{all}$)",
                       "median: entropy ($SE$, WPE, $S$, $T$)",
                       r"$\pm$ half-IQR (shaded in the panel colour)",
                       "storm main phase (onset to SYM-H minimum)"],
               fontsize=10.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.002), frameon=False)
    # In-figure title removed; the manuscript caption gives the full description.
    fig.subplots_adjust(hspace=0.22, wspace=0.20, top=0.96, bottom=0.115)
    out = RES / "e4_storm.png"
    fig.savefig(out, dpi=300)
    print("[save] %s" % out, flush=True)

    # ---- storm interval vs late recovery ----
    # Earlier versions compared h<0 with 0<=h<=6 under the labels "pre-storm" and
    # "main phase".  Two things were wrong: the 6-hour window misalignment meant h<0
    # sampled the post-minimum interval, and 0..+6 h is the early recovery, not the
    # main phase (which lies before the minimum).  The contrast below compares the
    # storm interval, where the broadband energy peaks (E_all max is exactly at h=0),
    # with the late recovery.  Entropy measures reach their minimum near +10..+22 h
    # and rise thereafter, so this is the interval pair in which the anti-correlation
    # with the band energy is expressed.
    print("\n=== storm interval (0..+12 h) vs late recovery (+36..+48 h) ===")
    m_storm = (h >= 0) & (h <= 12)
    m_late = (h >= 36) & (h <= 48)
    rows = []
    for c in cols:
        m, e = agg(c)
        a, b = np.nanmean(m[m_storm]), np.nanmean(m[m_late])
        rows.append({"quantity": c, "storm": a, "late_recovery": b,
                     "change_%": 100.0 * (b - a) / a if a else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "e4_storm_phase.csv", index=False)
    print(t.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
