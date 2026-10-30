#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""audit_e3.py -- Round-1 audit of the E3 "degeneracy-breaking" claim.

Suspicion.  §3.2 of the manuscript says the partial correlation
rho(metric, log Q_all | log E_Pc5) = -0.82 (SampEn) breaks the energy/dissipation
degeneracy only because Sigma_p is modulated by AE.  Two things must be checked:

  (1) Which Sigma_p did the batch runs actually use?  run_multiband.py defaults
      --kp to 0.0 and neither run_multiband_zw2.sh nor run_multiband_years.sh
      overrides it, in which case Sigma_p == 5.0 S everywhere and
      Q_all = dB_rms^2 / (mu0^2 * 5.0)  is a pure (unfiltered) disturbance energy.

  (2) If (1) holds, what is the -0.82 actually measuring?  Decompose:
        rho(metric, log Q_all | log E_Pc5)   <-- the reported number
        rho(metric, log Q_all | log E_all)   <-- residual = conductance term only
        rho(metric, log E_all | log E_Pc5)   <-- broadband energy beyond Pc5

      With constant Sigma_p, log Q_all = log E_all' + const, so the second line is
      identically 0 (if E_all' is the same quantity) and the first line reduces to
      a broadband-vs-narrowband comparison, NOT an energy-vs-dissipation test.

Outputs a table per station and the station means, plus the Sigma_p reconstruction.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

RES = "/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/" \
      "v8_entropy_dissipation/results"
MU0 = 4.0 * np.pi * 1e-7
METRICS = ["SE_full", "WPE", "SampEn", "Tsallis_q1.5", "CH_C", "IR_ordinal", "PE"]


def partial_spearman(x, y, z):
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 100:
        return np.nan, 0
    r = d.rank()
    rxy, rxz, ryz = r["x"].corr(r["y"]), r["x"].corr(r["z"]), r["y"].corr(r["z"])
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (float((rxy - rxz * ryz) / den) if den > 0 else np.nan), len(d)


def main():
    files = sorted(glob.glob(RES + "/mb_zw_*2015_w600_multiband.csv"))
    print("[audit] stations = %d" % len(files))

    # ---------- (1) recover the conductance actually used ----------
    print("\n=== (1) Sigma_p reconstruction from Q_all / dB_rms ===")
    print("    if Sigma_p were the constant 5.0 S, then Q_all == dB_rms^2/(mu0^2*5)")
    for f in files:
        st = f.split("mb_zw_")[1].split("2015")[0]
        d = pd.read_csv(f, usecols=["Q_all", "dB_rms", "E_Pc5"])
        ok = d["Q_all"].notna() & d["dB_rms"].notna() & (d["Q_all"] > 0)
        # implied Sigma_p:  S = dB^2/(mu0^2 Sigma)  ->  Sigma = dB_rms^2/(mu0^2 S)
        sig = (d.loc[ok, "dB_rms"].to_numpy() ** 2 * 1e-18) / (MU0 ** 2 * d.loc[ok, "Q_all"].to_numpy())
        print("  %-4s implied Sigma_p: median=%.4f  min=%.4f  max=%.4f  (n=%d)"
              % (st, np.median(sig), sig.min(), sig.max(), sig.size))

    # ---------- (2) decompose the reported partial correlation ----------
    print("\n=== (2) rho(metric, log Q_all | log E_Pc5) decomposition ===")
    recs = []
    for f in files:
        st = f.split("mb_zw_")[1].split("2015")[0]
        d = pd.read_csv(f)
        for c in ("Q_all", "E_Pc5", "E_all", "dB_rms"):
            if c not in d.columns:
                print("  [warn] %s missing %s" % (st, c))
        lq = np.log10(d["Q_all"].clip(1e-30))
        lp = np.log10(d["E_Pc5"].clip(1e-12))
        la = np.log10(d["E_all"].clip(1e-12)) if "E_all" in d.columns else np.nan
        ld = np.log10((d["dB_rms"] ** 2).clip(1e-12)) if "dB_rms" in d.columns else np.nan
        # how collinear are Q_all and dB_rms^2 ?
        m = d["Q_all"].notna() & d["dB_rms"].notna()
        rr = np.corrcoef(d.loc[m, "Q_all"], d.loc[m, "dB_rms"].to_numpy() ** 2)[0, 1]
        rec = {"station": st, "corr(Q_all, dB_rms^2)": round(float(rr), 6)}
        for k in METRICS:
            if k not in d.columns:
                continue
            v = d[k]
            rec["%s|Epc5" % k] = partial_spearman(v, lq, lp)[0]
            if np.isscalar(la):
                rec["%s|Eall" % k] = partial_spearman(v, lq, la)[0]
                rec["%s|dB" % k] = partial_spearman(v, lq, ld)[0]
        recs.append(rec)

    t = pd.DataFrame(recs).set_index("station")
    pd.set_option("display.width", 200)
    print("\n-- per station --")
    print(t.round(3).to_string())

    print("\n-- station means --")
    for k in METRICS:
        c1, c2, c3 = ("%s|Epc5" % k), ("%s|Eall" % k), ("%s|dB" % k)
        if c1 not in t.columns:
            continue
        line = "  %-14s |Epc5: %+.3f" % (k, t[c1].mean())
        if c2 in t.columns:
            line += "   |Eall: %+.3f" % t[c2].mean()
        if c3 in t.columns:
            line += "   |dB_rms^2: %+.3f" % t[c3].mean()
        print(line)
    print("\n  corr(Q_all, dB_rms^2) mean = %.8f  (1.0 => Q_all is a pure energy)"
          % t["corr(Q_all, dB_rms^2)"].mean())

    t.to_csv(RES + "/audit_e3_decomposition.csv")
    print("\n[save] audit_e3_decomposition.csv")


if __name__ == "__main__":
    main()
