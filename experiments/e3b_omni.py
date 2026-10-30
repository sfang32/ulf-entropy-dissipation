"""e3b_omni.py -- E3b: cross-validate the entropy-dissipation link with GENUINELY
INDEPENDENT upstream/ionospheric quantities (not derived from the ground field itself).

E3 showed entropy keeps strong partial correlation with Q_all after removing Pc5 power.
But Q_all is computed from the same ground series -> risk of "same-source" correlation.
Here we test against two independent chains:

  1. solar-wind driver power   Q_sw = (V * Bz_south)^2      [ (km/s * nT)^2 ]
     (proportional to the square of the convection electric field; no ground data)
  2. AE index                  (auroral electrojet -> Joule heating proxy)
  3. SYM-H                     (ring current, for contrast)

For each station we merge the 10-min OMNI aggregates onto the multiband window grid and
report raw + partial Spearman (controlling log E_Pc5, and controlling log Q_all).

Run on the server: python e3b_omni.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cdflib

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
OMNI = Path("/data/ftp/fsf/databases/data_omni/omni_1min/2015")
MLAT = {"MHT": 48.36, "MZL": 44.54, "NAT": 38.44, "SSL": 35.04, "MLS": 29.09,
        "PXT": 25.47, "JFT": 24.85, "LAT": 23.66, "SYS": 13.16, "SYT": 21.41}
METRICS = ["SE_full", "WPE", "SampEn", "Tsallis_q1.5", "CH_C", "IR_ordinal"]


def _var(c, name):
    """Read a CDF variable as float with fill values -> NaN.

    The server's cdflib is old and does not accept the fillval= keyword.
    """
    x = np.asarray(c.varget(name), dtype=float)
    return np.where(np.abs(x) > 1e30, np.nan, x)


def load_omni_2015():
    frames = []
    for f in sorted(glob.glob(str(OMNI / "omni_hro_1min_2015*_v01.cdf"))):
        c = cdflib.CDF(f)
        t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
        d = pd.DataFrame({
            "bz": _var(c, "BZ_GSM"),
            "v": _var(c, "flow_speed"),
            "ae": _var(c, "AE_INDEX"),
            "symh": _var(c, "SYM_H"),
        }, index=t)
        frames.append(d)
    omni = pd.concat(frames).sort_index()
    omni = omni[~omni.index.duplicated()]
    # southward Bz -> convection electric field proxy; Q_sw ~ |E_conv|^2
    bzs = np.clip(-omni["bz"], 0, None)
    omni["Q_sw"] = (omni["v"] * bzs) ** 2
    # 10-min means aligned to window starts (mb windows start at :00/:10/:20...)
    agg = omni.resample("10min").mean()
    agg["time"] = agg.index
    return agg


def partial(x, y, z):
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 100:
        return np.nan
    r = d.rank()
    rxy, rxz, ryz = r["x"].corr(r["y"]), r["x"].corr(r["z"]), r["y"].corr(r["z"])
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return np.nan if den <= 0 else float((rxy - rxz * ryz) / den)


def main():
    omni = load_omni_2015()
    print("[omni] 10-min aggregates: %d  (%s .. %s)"
          % (len(omni), omni.index[0], omni.index[-1]), flush=True)

    # Phase-I 2015 canonical 10-min (w600) runs, 9 stations (year pinned: a looser
    # glob would mix in the 2011-2022 multi-year batch)
    files = sorted(glob.glob(str(RES / "mb_zw_*2015_w600_multiband.csv")))
    if len(files) != 9:
        print("[e3b] WARNING: expected 9 phase-I 2015 stations, found %d" % len(files),
              flush=True)
    rows = []
    for f in files:
        st = os.path.basename(f).split("_")[2].split("2015")[0]
        df = pd.read_csv(f)
        if "E_Pc5" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        df["time"] = df["time"].dt.floor("10min")
        m = df.merge(omni[["time", "Q_sw", "ae", "symh"]], on="time", how="inner")
        m = m[m["E_Pc5"].notna() & m["WPE"].notna()]
        if len(m) < 500:
            print("  [skip] %s: only %d merged rows" % (st, len(m)), flush=True)
            continue
        le = np.log10(m["E_Pc5"].clip(1e-12))
        lq = np.log10(m["Q_all"].clip(1e-30))
        for k in METRICS:
            if k not in m.columns:
                continue
            v = m[k]
            rec = {"station": st, "mlat": MLAT.get(st, np.nan), "metric": k,
                   "n": len(m),
                   "rho_Qsw": float(v.corr(np.log10(m["Q_sw"].clip(1e-6)),
                                           method="spearman")),
                   "rho_AE": float(v.corr(np.log10(m["ae"].clip(1)),
                                          method="spearman")),
                   "rho_SYMH": float(v.corr(m["symh"], method="spearman")),
                   "pQsw_given_Epc5": partial(v, np.log10(m["Q_sw"].clip(1e-6)), le),
                   "pAE_given_Epc5": partial(v, np.log10(m["ae"].clip(1)), le),
                   "pQsw_given_Qall": partial(v, np.log10(m["Q_sw"].clip(1e-6)), lq),
                   "pAE_given_Qall": partial(v, np.log10(m["ae"].clip(1)), lq)}
            rows.append(rec)
    tab = pd.DataFrame(rows)
    out = RES / "e3b_omni.csv"
    tab.to_csv(out, index=False)
    print("[save] %s" % out, flush=True)

    for col, lab in [("rho_Qsw", "raw  rho(metric, log Q_sw)"),
                     ("rho_AE", "raw  rho(metric, log AE)"),
                     ("rho_SYMH", "raw  rho(metric, SYM-H)"),
                     ("pQsw_given_Epc5", "PARTIAL rho(metric, log Q_sw | log E_Pc5)"),
                     ("pAE_given_Epc5", "PARTIAL rho(metric, log AE | log E_Pc5)"),
                     ("pQsw_given_Qall", "PARTIAL rho(metric, log Q_sw | log Q_all)")]:
        piv = tab.pivot_table(index="metric", columns="station", values=col)
        piv = piv.reindex(index=[m for m in METRICS if m in piv.index])
        piv["MEAN"] = piv.mean(axis=1)
        print("\n=== %s ===" % lab)
        print(piv.round(3).to_string(), flush=True)

    print("\n=== E3b VERDICT (mean |partial| over stations, controlling E_Pc5) ===")
    piv = tab.pivot_table(index="metric", columns="station", values="pQsw_given_Epc5")
    piv2 = tab.pivot_table(index="metric", columns="station", values="pAE_given_Epc5")
    for m in [x for x in METRICS if x in piv.index]:
        a, b = piv.loc[m].mean(), piv2.loc[m].mean()
        print("  %-12s  Q_sw|Epc5: %+.3f (%s)   AE|Epc5: %+.3f (%s)"
              % (m, a, "PASS" if abs(a) >= 0.2 else "fail",
                 b, "PASS" if abs(b) >= 0.2 else "fail"))


if __name__ == "__main__":
    main()
