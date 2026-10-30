"""v8 prototype v2: entropy vs Pc5-band energy vs dissipation rate.

Per 方老师 (2026-09-30): do NOT use the third-party Pc5 event labels.
Extract the Pc5 band energy ourselves, following the tool_cor methodology
(research_ulf/_submit/submit_yuanxinxihua/tool_cor):
  * get_Wn():      Pc5 = 1.6-6.7 mHz (Pc3 23-100, Pc4 6.7-23 mHz)
  * band_filter(): 10th-order Butterworth sos bandpass (scipy.signal.sosfilt)
The daily continuous 1 Hz H series is bandpassed ONCE per day, then cut into
10-min windows: entropy battery on the raw window; Pc5 band energy and the
Pc5-band Poynting dissipation rate on the band-limited window.

Per-window chain (same band, self-consistent):
    entropy battery  vs  E_pc5 [nT^2]  vs  P_pc5 [W/m^2]
Reference full-band quantities kept: dB_rms, Q_all.

Run:
  /data/ssd-stor/fsf/anaconda3/bin/python run_prototype.py --station MHT --year 2015
  (smoke test: append --max-days 2)
"""
from __future__ import annotations
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import entropy_battery as eb
import deposition_rate as dr

ZW = Path("/data/ssd-stor/fsf/databases/data_ulf/dataset_pc/ZW")
# AACGM-v2 magnetic latitude (aacgmv2 2.7 via server torch env, ref 2015-07-01 00:00 UT;
# table: v8_entropy_dissipation/data/stations_v8_aacgm.csv). SSL recovered 2026-09-30.
# NOTE: these differ from v6's dipole mlat by +3.5..+4.7 deg at these longitudes;
# AACGM is the correct stratifier (cf. project_ts paper-2). Sigma_p ignores mlat for now.
MLAT = {"MHT": 48.4, "MZL": 44.5, "NAT": 38.4, "SSL": 35.0, "MLS": 29.1,
        "JFT": 24.8, "HZT": 24.4, "PXT": 25.5, "LAT": 23.7, "SYS": 13.2, "SYT": 21.4}

# tool_cor/core/utils.py get_Wn() band edges (Hz)
WN = {"Pc1": (200e-3, 999e-3), "Pc2": (100e-3, 200e-3), "Pc3": (23e-3, 100e-3),
      "Pc4": (6.7e-3, 23e-3), "Pc5": (1.6e-3, 6.7e-3), "Pi2": (6.7e-3, 25e-3)}

WIN = 600     # 10-min @ 1 Hz
EDGE = 1800   # discard 30 min at both day edges (bandpass transient)
SCALAR_KEYS = ["SE_full", "SE_Pc3", "SE_Pc4", "SE_Pc5", "PE", "WPE",
               "SampEn", "Tsallis_q1.5", "CH_H", "CH_C", "SVD", "IR_ordinal"]


def find_h_column(df):
    cols = {c.upper(): c for c in df.columns}
    for key in ("H", "DH", "HORIZONTAL"):
        if key in cols:
            return cols[key]
    for c in df.columns:
        cu = c.upper()
        if "H" in cu and "TIME" not in cu and "DATE" not in cu:
            return c
    raise KeyError(f"cannot find H column in {list(df.columns)}")


def band_filter(x, band="Pc5", fs=1.0):
    """10th-order Butterworth sos bandpass, as tool_cor/core/tfa.py."""
    x = np.asarray(x, dtype=float)
    sos = sps.butter(N=10, Wn=WN[band], btype="bp", fs=fs, output="sos")
    return sps.sosfilt(sos, x - np.nanmean(x))


def build_daily(files):
    """Group 10-min files into days by filename timestamp."""
    days = OrderedDict()
    for f in files:
        try:
            t = pd.to_datetime(f.stem)
        except Exception:
            continue
        days.setdefault(t.date(), []).append((t, f))
    return days


def daily_series(flist):
    """Concatenate a day's 10-min csv files onto a regular 1 s grid.

    Returns (series, n_raw_points) or (None, 0) if coverage is too low.
    """
    ts, hs = [], []
    for t, f in sorted(flist, key=lambda x: x[0]):
        df = pd.read_csv(f)
        hs.append(df[find_h_column(df)].to_numpy(dtype=float))
        ts.append(t)
    if not hs:
        return None, 0
    h_cat = np.concatenate(hs)
    t_cat = []
    for t, h in zip(ts, hs):
        t_cat.append(pd.date_range(start=t, periods=len(h), freq="s"))
    t_cat = t_cat[0].append(t_cat[1:]) if len(t_cat) > 1 else t_cat[0]
    s = pd.Series(h_cat, index=t_cat)
    s = s[~s.index.duplicated()]
    n_raw = s.notna().sum()
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="s"))
    s = s.interpolate(limit_direction="both")
    if n_raw < 0.5 * len(s):     # less than half a day present -> skip
        return None, n_raw
    return s, n_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="MHT")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--m-pe", type=int, default=4)
    ap.add_argument("--max-days", type=int, default=0,
                    help="smoke test: only first N days (0 = all)")
    ap.add_argument("--kp", type=float, default=0.0,
                    help="constant Kp for Sigma_p (OMNI modulation: TODO)")
    ap.add_argument("--outdir",
                    default="/data/ssd-stor/fsf/workspaces/research_ulf/"
                             "ground_pc5/v8_entropy_dissipation/results")
    args = ap.parse_args()

    base = ZW / args.station / str(args.year) / "10min" / "feature_1Hz"
    files = sorted(base.rglob("*.csv"))
    if not files:
        raise SystemExit(f"no feature_1Hz csv under {base}")
    days = build_daily(files)
    day_items = list(days.items())
    if args.max_days:
        day_items = day_items[:args.max_days]
    print(f"[proto] {len(day_items)} days for {args.station} {args.year}")

    mlat = MLAT.get(args.station, 30.0)
    rows = []
    for dnum, (day, flist) in enumerate(day_items, 1):
        s, n_raw = daily_series(flist)
        if s is None:
            print(f"[proto] skip {day}: coverage too low ({n_raw} pts)")
            continue
        raw = s.to_numpy()
        filt = band_filter(raw, band="Pc5", fs=1.0)
        sig = dr.pedersen_conductance(mlat, args.kp)

        for w0 in range(0, raw.size - WIN + 1, WIN):
            rw = raw[w0:w0 + WIN]
            if np.all(~np.isfinite(rw)):
                continue
            S_all, E_all, dB = dr.window_deposition(rw, mlat, kp=args.kp, fs=1.0)
            ent = eb.entropy_battery(rw, fs=1.0, m_pe=args.m_pe)
            row = {"time": s.index[w0], "day": str(day),
                   "Q_all": S_all, "E_int_all": E_all, "dB_rms": dB}
            if (w0 < EDGE) or (w0 + WIN + EDGE > raw.size):
                row["E_pc5"] = np.nan      # bandpass transient zone
                row["P_pc5"] = np.nan
            else:
                fw = filt[w0:w0 + WIN]
                row["E_pc5"] = float(np.mean(fw ** 2))            # nT^2
                row["P_pc5"] = float(np.mean(                     # W/m^2
                    dr.poynting_flux(fw * 1e-9, sig)))
            for k in SCALAR_KEYS:
                row[k] = ent.get(k, np.nan)
            rows.append(row)
        if dnum % 30 == 0:
            print(f"[proto] ... day {dnum}/{len(day_items)}")

    out = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"v2_{args.station}{args.year}_pc5energy.csv"
    out.to_csv(csv_path, index=False)
    n_valid = int(out["E_pc5"].notna().sum())
    print(f"[proto] wrote {csv_path}  ({len(out)} windows, {n_valid} in-band)")

    # ---- stats: entropy vs same-band energy / same-band dissipation ----
    ok = out["E_pc5"].notna() & out["WPE"].notna()
    if ok.sum() > 30:
        sub = out[ok]
        le = np.log10(sub["E_pc5"].clip(1e-12))
        lq5 = np.log10(sub["P_pc5"].clip(1e-30))
        lqa = np.log10(sub["Q_all"].clip(1e-30))
        print(f"[stats] E_pc5[nT^2]: median={sub['E_pc5'].median():.3g} "
              f"p90={sub['E_pc5'].quantile(0.9):.3g}")
        print(f"[stats] P_pc5[W/m^2]: median={sub['P_pc5'].median():.3g}")
        for key in ["SE_full", "SE_Pc5", "PE", "WPE", "CH_C", "SVD",
                    "IR_ordinal"]:
            k = sub[key]
            if k.notna().sum() > 30:
                print(f"[stats] {key}: "
                      f"(vs logE_pc5)={k.corr(le, method='spearman'):.3f} "
                      f"(vs logP_pc5)={k.corr(lq5, method='spearman'):.3f} "
                      f"(vs logQ_all)={k.corr(lqa, method='spearman'):.3f}")

    # ---- figure: entropy vs self-extracted Pc5 band energy ----
    if ok.sum() > 10:
        sub = out[ok]
        plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                             "xtick.labelsize": 10, "ytick.labelsize": 10,
                             "axes.titlesize": 11})
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        xlo = float(sub["E_pc5"].min())
        xhi = float(sub["E_pc5"].max())
        for ax, key in zip(axes.ravel(),
                           ["SE_full", "WPE", "CH_C", "IR_ordinal"]):
            m = sub[key].notna()
            ax.scatter(sub.loc[m, "E_pc5"], sub.loc[m, key],
                       s=9, alpha=0.35, c="steelblue", edgecolors="none")
            ax.set_xscale("log")
            ax.set_xlim(xlo * 0.7, xhi * 1.5)
            ax.set_xlabel(r"Pc5 band energy $E_{Pc5}$ [nT$^2$]")
            ax.set_ylabel(key)
            r = sub.loc[m, key].corr(np.log10(sub.loc[m, "E_pc5"]),
                                     method="spearman")
            ax.set_title(f"{key} vs E_pc5   Spearman={r:.3f}")
            ax.grid(alpha=0.25, linewidth=0.4)
        fig.suptitle(f"{args.station} {args.year}: entropy vs Pc5-band energy "
                     "(self-extracted, tool_cor method)", fontsize=12)
        fig.subplots_adjust(hspace=0.36, wspace=0.26, top=0.90)
        fig.savefig(outdir / f"v2_energy_{args.station}{args.year}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
