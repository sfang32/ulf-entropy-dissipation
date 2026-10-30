"""e7_anchor.py -- E7: absolute anchoring of the ground-based dissipation estimate.

The pipeline estimates the Pc5-band downward Poynting flux
        P_pc5 = mean( dB_pc5^2 / (mu0^2 Sigma_p) )   [W/m^2]
with a physics-driven Sigma_p (E2). E7 asks whether the ABSOLUTE values are credible,
by comparing the per-station distribution against the ISR-measured ULF ionospheric
Poynting-flux range reported by van Hazendonk et al. (2024, JGR e2024JA032633)
and Rae et al. (2007).

Reported reference band (ULF / FLR-driven, high-latitude ISR): ~1e-8 .. 1e-6 W/m^2.
We report per-station median / p90 / p99 / max and the fraction inside that band.

Run on the server: python e7_anchor.py
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

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
OMNI = Path("/data/ftp/fsf/databases/data_omni/omni_1min/2015")
MU0 = 4.0 * np.pi * 1e-7
REF_LO, REF_HI = 1e-8, 1e-6          # van Hazendonk 2024 / Rae 2007 ULF range [W/m^2]
MLAT = {"MHT": 48.36, "MZL": 44.54, "NAT": 38.44, "SSL": 35.04, "MLS": 29.09,
        "PXT": 25.47, "JFT": 24.85, "LAT": 23.66, "SYS": 13.16}


def _var(c, name):
    x = np.asarray(c.varget(name), dtype=float)
    return np.where(np.abs(x) > 1e30, np.nan, x)


def load_ae():
    fr = []
    for f in sorted(glob.glob(str(OMNI / "omni_hro_1min_2015*_v01.cdf"))):
        c = cdflib.CDF(f)
        t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
        fr.append(pd.Series(_var(c, "AE_INDEX"), index=t, name="ae"))
    s = pd.concat(fr).sort_index()
    s = s[~s.index.duplicated()]
    a = s.resample("10min").mean().to_frame()
    a["time"] = a.index
    return a


def main():
    ae = __import__("omni_2015_ascii").load_ae_10min()  # patched: OL-CB ASCII OMNI (fast) instead of stalled /data/ftp CDF
    rows, dist = [], {}
    # pinned to 2015: results/ also holds the 2011-2022 multi-year batch, and a
    # bare `mb_zw_*_w600_multiband.csv` glob would silently turn every year into
    # an extra "station" whose name parses as e.g. MHT2013 and whose mlat is NaN
    files = sorted(glob.glob(str(RES / "mb_zw_*2015_w600_multiband.csv")))
    assert len(files) == 9, \
        "expected the nine 2015 Phase-I stations, found %d" % len(files)
    for f in files:
        st = os.path.basename(f).split("_")[2].split("2015")[0]
        df = pd.read_csv(f)
        if "E_Pc5" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None) \
            .dt.floor("10min")
        m = df.merge(ae, on="time", how="inner")
        m = m[m["E_Pc5"].notna()]
        m = m[m["E_Pc5"] <= 1e4]                       # drop saturation-contaminated
        if len(m) < 500:
            continue
        sigma = 5.0 * (m["ae"].clip(1) / 100.0) ** 0.5   # same driver as E2 "AE"
        # UNIT TRAP: E_pc5 is in nT^2 -> must convert to T^2 (1e-18) BEFORE dividing
        # by mu0^2. (run_multiband.py does this correctly via fw*1e-9.)
        P = m["E_Pc5"] * 1e-18 / (MU0 ** 2 * sigma)      # W/m^2
        P = P.to_numpy()
        dist[st] = P
        inside = np.mean((P >= REF_LO) & (P <= REF_HI)) * 100
        rows.append({"station": st, "mlat": MLAT.get(st, np.nan), "n": len(P),
                     "median": np.nanmedian(P),
                     "p90": np.nanpercentile(P, 90),
                     "p99": np.nanpercentile(P, 99),
                     "max": np.nanmax(P), "pct_in_ref_band": inside})
    tab = pd.DataFrame(rows).sort_values("mlat", ascending=False)
    out = RES / "e7_anchor.csv"
    tab.to_csv(out, index=False)
    print("[save] %s" % out, flush=True)
    pd.set_option("display.float_format", lambda v: "%.3g" % v)
    print(tab.to_string(index=False), flush=True)

    print("\n=== anchor check ===", flush=True)
    print("reference ULF Poynting range (van Hazendonk 2024 / Rae 2007): "
          "%.0e .. %.0e W/m^2" % (REF_LO, REF_HI), flush=True)
    med = np.nanmedian([np.nanmedian(v) for v in dist.values()])
    print("median of per-station medians = %.3g W/m^2 -> %s"
          % (med, "inside reference band"
             if REF_LO <= med <= REF_HI else
             ("below band (%.1f dex)" % np.log10(med / REF_LO) if med < REF_LO
              else "above band (%.1f dex)" % np.log10(med / REF_HI))), flush=True)

    # ---- figure: P_pc5 distributions vs reference band ----
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    order = tab["station"].tolist()
    # Drop physically implausible values.  P ~ 1e-27 W/m^2 is ~20 dex below the ISR
    # range and comes from near-quiet windows where dB_pc5^2 underflows the estimate;
    # keeping them collapsed the whole y axis and hid every real distribution.
    FLOOR = 1e-12
    data, n_drop, n_keep = [], 0, 0
    for s in order:
        v = dist[s][np.isfinite(dist[s])]
        n_drop += int((v <= FLOOR).sum())
        v = v[v > FLOOR]
        n_keep += v.size
        data.append(v)
    print("[e7] dropped %d/%d windows below %.0e W/m^2 (%.2f%%)"
          % (n_drop, n_drop + n_keep, FLOOR, 100.0 * n_drop / max(1, n_drop + n_keep)),
          flush=True)

    bp = ax.boxplot(data, vert=True, showfliers=False, patch_artist=True,
                    widths=0.6, medianprops=dict(color="k", lw=1.5))
    for p in bp["boxes"]:
        p.set(facecolor="#9ecae1", alpha=0.8, edgecolor="0.3")
    ax.axhspan(REF_LO, REF_HI, color="#d62728", alpha=0.15, lw=0,
               label="ISR reference range (van Hazendonk 2024 / Rae 2007)")
    ax.axhline(med, color="k", ls="--", lw=1.0,
               label=r"median %.1e W m$^{-2}$ (%.1f dex below band)"
                     % (med, np.log10(med / REF_LO)))
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 2e-4)
    ax.set_xticklabels(["%s\n%.0f$^\\circ$" % (s, MLAT.get(s, np.nan)) for s in order])
    ax.set_ylabel(r"$P_{\rm Pc5}$ [W m$^{-2}$]")
    ax.set_xlabel("station (AACGM magnetic latitude)")
    # Title removed; the manuscript caption describes the panel.
    ax.grid(alpha=0.25, axis="y", lw=0.4)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(RES / "e7_anchor.png", dpi=300)
    print("[save] e7_anchor.png", flush=True)


if __name__ == "__main__":
    main()
