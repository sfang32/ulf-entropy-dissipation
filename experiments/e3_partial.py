"""e3_partial.py -- E3 "death-or-life" test: does entropy carry information BEYOND power?

v6 admitted its entropy signature is "largely shared with band-limited power".
E3 asks: after controlling for the band energy, does the entropy still correlate with
an independent dissipative quantity?

NOTE on P_pc5: with a constant Sigma_p, P_band is a monotone function of E_band, so a
partial correlation against P_band is degenerate. We therefore use the full-band
Poynting flux Q_all (a genuinely different quantity) plus two directions:

  (a) rho(metric, log Q_all | log E_Pc5)   -- entropy beyond Pc5 power?
  (b) rho(metric, log E_Pc5 | log Q_all)   -- Pc5 power beyond total disturbance?

Partial Spearman = Pearson on ranks.
Criterion (as designed in v8_experiments.md): |rho_partial| >= 0.2 for at least one
metric => H survives; all ~0 => the claim degrades to "entropy is a power proxy".

Run on the server: python e3_partial.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
MLAT = {"MHT": 48.36, "MZL": 44.54, "NAT": 38.44, "SSL": 35.04, "MLS": 29.09,
        "PXT": 25.47, "JFT": 24.85, "LAT": 23.66, "SYS": 13.16, "SYT": 21.41}
METRICS = ["SE_full", "WPE", "SampEn", "SVD", "Tsallis_q1.5", "CH_C",
           "IR_ordinal", "PE"]
BAND = "Pc5"


def partial_spearman(x, y, z):
    """rho(x, y | z) using ranks -> Pearson partial correlation."""
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 100:
        return np.nan, 0
    r = d.rank()
    rxy = r["x"].corr(r["y"])
    rxz = r["x"].corr(r["z"])
    ryz = r["y"].corr(r["z"])
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if den <= 0:
        return np.nan, len(d)
    return float((rxy - rxz * ryz) / den), len(d)


def main():
    # Phase-I 2015 canonical 10-min (w600) runs, 9 stations.
    # NOTE the year is pinned: results/ also holds the 2011-2022 multi-year batch,
    # and a looser glob silently mixes 106 files from 12 different years.
    files = sorted(glob.glob(str(RES / "mb_zw_*2015_w600_multiband.csv")))
    if len(files) != 9:
        print("[e3] WARNING: expected 9 phase-I 2015 stations, found %d: %s"
              % (len(files), [os.path.basename(f) for f in files]), flush=True)
    if not files:
        raise SystemExit("no baseline mb results (10-min) found")
    rows = []
    for f in files:
        st = os.path.basename(f).split("_")[2].split("2015")[0]
        df = pd.read_csv(f)
        need = [f"E_{BAND}", "Q_all", "WPE"]
        if any(c not in df.columns for c in need):
            continue
        s = df[df[f"E_{BAND}"].notna() & df["WPE"].notna()].copy()
        le = np.log10(s[f"E_{BAND}"].clip(1e-12))
        lq = np.log10(s["Q_all"].clip(1e-30))
        for k in METRICS:
            if k not in s.columns:
                continue
            v = s[k]
            pa, n = partial_spearman(v, lq, le)      # entropy | Pc5 power
            pb, _ = partial_spearman(v, le, lq)      # Pc5 power | total
            rows.append({"station": st, "mlat": MLAT.get(st, np.nan), "metric": k,
                         "rho_vs_Q": float(v.corr(lq, method="spearman")),
                         "rho_vs_Epc5": float(v.corr(le, method="spearman")),
                         "partial_Q_given_Epc5": pa,
                         "partial_Epc5_given_Q": pb, "n": n})
    tab = pd.DataFrame(rows)
    out = RES / "e3_partial.csv"
    tab.to_csv(out, index=False)
    print("[save] %s" % out)

    piv = tab.pivot_table(index="metric", columns="station",
                          values="partial_Q_given_Epc5")
    order = [m for m in METRICS if m in piv.index]
    piv = piv.reindex(index=order)
    print("\n=== PARTIAL Spearman: rho(metric, log Q_all | log E_Pc5), per station ===")
    show = piv.copy()
    show["MEAN"] = piv.mean(axis=1)
    show["|MEAN|"] = piv.mean(axis=1).abs()
    print(show.round(3).to_string())
    print("\n(control: band energy removed. |MEAN| >= 0.2 => entropy carries "
          "independent information => H survives)")

    piv2 = tab.pivot_table(index="metric", columns="station",
                           values="partial_Epc5_given_Q")
    piv2 = piv2.reindex(index=order)
    print("\n=== PARTIAL Spearman: rho(metric, log E_Pc5 | log Q_all) ===")
    show2 = piv2.copy()
    show2["MEAN"] = piv2.mean(axis=1)
    print(show2.round(3).to_string())

    # ---- verdict per metric ----
    print("\n=== E3 VERDICT (mean over stations of |partial| for Q|E) ===")
    for m in order:
        v = piv.loc[m].dropna()
        if v.empty:
            continue
        mu = v.mean()
        n_same = int((np.sign(v) == np.sign(mu)).sum())
        flag = "PASS" if abs(mu) >= 0.2 else "fail"
        print("  %-12s mean=%+.3f  same-sign %d/%d  -> %s"
              % (m, mu, n_same, len(v), flag))


if __name__ == "__main__":
    main()
