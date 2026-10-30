"""e2_sigma.py -- E2/E6: does a physically modulated Pedersen conductance change the
entropy-dissipation calibration?

With a CONSTANT Sigma_p, P_band = E_band / (mu0^2 Sigma_p) is a monotone function of
E_band, so the two are statistically indistinguishable (their Spearman rho with any
metric is IDENTICAL). E2 exists to break that degeneracy.

Available drivers: the OMNI 1-min HRO files carry NO Kp -- only AE/AL/AU, SYM-H and
PC_N_INDEX. We therefore use three schemes (E6 = sensitivity across schemes):

  S1 const      Sigma_p = 5.0 S                        (baseline, current pipeline)
  S2 AE-driven  Sigma_p = 5.0 * (AE/100)^0.5           (electrojet ~ Sigma_p * E)
  S3 PC-driven  Sigma_p = 5.0 * (PC_N/1.0)^0.5         (polar-cap index ~ conductance)

For each station/scheme we report rho(metric, log E_band), rho(metric, log P_band) and
the partial rho(metric, log P_band | log E_band) -- the last one is what E3 could not
test under a constant Sigma_p.

Run on the server: python e2_sigma.py
"""
from __future__ import annotations

import argparse

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
METRICS = ["SE_full", "WPE", "SampEn", "Tsallis_q1.5", "CH_C", "IR_ordinal"]


def _var(c, name):
    """Read a CDF variable and mask its fill values.

    NOTE: a guard of |x| > 1e30 alone is NOT enough for OMNI.  The AE/PC time
    series use sentinel fills around 999999.9, which passes that test and then
    becomes a *constant* conductance -> log P becomes an affine function of log E
    -> the partial correlation collapses to exactly 0.  That failure is silent:
    it looks like a null result rather than a data problem (observed for the
    2020-2022 mirror files before this fix).  Prefer the variable's own FILLVAL;
    fall back to a threshold that is far above any physical value of AE/PC/SYM-H.
    """
    x = np.asarray(c.varget(name), dtype=float)
    fill = None
    try:
        attrs = c.varattsget(name)
        for k in ("FILLVAL", "FILL_VALUE"):
            if k in attrs:
                fill = float(np.asarray(attrs[k]).ravel()[0])
                break
    except Exception:
        fill = None
    bad = ~np.isfinite(x) | (np.abs(x) > 1e30)
    if fill is not None and np.isfinite(fill):
        bad |= np.isclose(x, fill, rtol=1e-6, atol=0.0)
    bad |= np.abs(x) >= 9.9e4          # sentinel fills (99999.9 / 999999.9)
    return np.where(bad, np.nan, x)


def load_drivers(year=2015, root=None):
    """OMNI 1-min drivers for one year, aggregated to 10-min means.

    root overrides the mirror that OMNI's own parent directory implies -- needed
    when the canonical mirror carries fill values in the AE column (2021-2022)
    and a freshly downloaded copy is used instead.
    """
    root = Path(root) if root else OMNI.parent
    frames = []
    pat = str(root / str(year) / ("omni_hro_1min_%d*_v01.cdf" % year))
    for f in sorted(glob.glob(pat)):
        c = cdflib.CDF(f)
        t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
        frames.append(pd.DataFrame({
            "ae": _var(c, "AE_INDEX"),
            "pc": _var(c, "PC_N_INDEX"),
            "symh": _var(c, "SYM_H"),
        }, index=t))
    d = pd.concat(frames).sort_index()
    d = d[~d.index.duplicated()]
    agg = d.resample("10min").mean()
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


SCHEMES = {"const": lambda d: np.full(len(d), 5.0),
           "AE": lambda d: 5.0 * (d["ae"].clip(1) / 100.0) ** 0.5,
           "PC": lambda d: 5.0 * (d["pc"].clip(0.1) / 1.0) ** 0.5}


def station_of(bn):
    """MHT from mb_zw_MHT2015_w600_multiband.csv; year-agnostic."""
    import re
    m = re.search(r"mb_[a-z0-9]{2,3}_([A-Za-z0-9]+?)(\d{4})_", bn)
    return m.group(1) if m else bn.split("_")[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="zw", choices=["zw", "zwraw", "zw2", "sm"])
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--omni-root", default=None,
                    help="override the OMNI 1-min root directory")
    a = ap.parse_args()
    drv = load_drivers(a.year, a.omni_root)
    print("[omni] %d 10-min rows (year %d); Sigma_p ranges:"
          % (len(drv), a.year), flush=True)
    for k, fn in SCHEMES.items():
        s = fn(drv)
        print("   %-5s %.2f .. %.2f S (median %.2f)" % (k, np.nanmin(s), np.nanmax(s),
                                                        np.nanmedian(s)), flush=True)

    # canonical runs only, with the source and year pinned (see mb_summary.py):
    # a looser glob mixes in the 2011-2022 batch and the window-sensitivity runs.
    pat = "mb_%s_*%d_w600_multiband.csv" % (a.src, a.year)
    files = sorted(glob.glob(str(RES / pat)))
    exp = 9 if a.src == "zw" else 28
    print("[e2] src=%s year=%d -> %d files (expected %d)"
          % (a.src, a.year, len(files), exp), flush=True)
    rows = []
    for f in files:
        st = station_of(os.path.basename(f))
        df = pd.read_csv(f)
        if "E_Pc5" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.floor("10min")
        m = df.merge(drv, on="time", how="inner")
        m = m[m["E_Pc5"].notna() & m["WPE"].notna()]
        if len(m) < 500:
            continue
        for band in ("Pc5", "all"):
            e = m[f"E_{band}"]
            le = np.log10(e.clip(1e-12))
            for sch, fn in SCHEMES.items():
                sig = fn(m)
                lp = np.log10((e / (MU0 ** 2 * sig)).clip(1e-30))   # P_band [W/m^2]
                for k in METRICS:
                    if k not in m.columns:
                        continue
                    v = m[k]
                    rows.append({"station": st, "band": band, "scheme": sch,
                                 "metric": k, "n": len(m),
                                 "rho_E": float(v.corr(le, method="spearman")),
                                 "rho_P": float(v.corr(lp, method="spearman")),
                                 "p_P_given_E": partial(v, lp, le)})
    tab = pd.DataFrame(rows)
    # year-tagged so the per-station values survive: the figure that summarises
    # every year needs them to form the across-station confidence interval, and a
    # single un-tagged e2_sigma.csv is overwritten by each run
    out = RES / ("e2_sigma_%s%d.csv" % (a.src, a.year))
    tab.to_csv(out, index=False)
    print("\n[save] %s" % out, flush=True)

    for band in ("Pc5", "all"):
        sub = tab[tab.band == band]
        print("\n=== band %s : rho(metric, log P_band | log E_band) ===" % band, flush=True)
        piv = sub.pivot_table(index="metric", columns="scheme", values="p_P_given_E",
                              aggfunc="mean")
        piv = piv.reindex(index=[m for m in METRICS if m in piv.index])
        print(piv.round(3).to_string(), flush=True)
        print("   (rho_E vs rho_P difference by scheme, mean over states):", flush=True)
        for sch in SCHEMES:
            s2 = sub[sub.scheme == sch]
            print("     %-5s  E: %+.3f   P: %+.3f" %
                  (sch, s2["rho_E"].mean(), s2["rho_P"].mean()), flush=True)

    # ---- figure: the E-vs-P degeneracy, and how Sigma_p breaks it ----
    # NOTE: the headline quantity is p_P_given_E (partial rho controlling the band
    # energy), NOT the raw rho_P.  Under a constant Sigma_p, P is a monotone function
    # of E, so that partial correlation is undefined by construction -- which is
    # exactly the degeneracy E2 is designed to break.
    sub = tab[tab.band == "Pc5"]
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})

    fig, ax = plt.subplots(1, 2, figsize=(13.8, 5.4))
    order = [m for m in METRICS if m in set(sub["metric"])]
    y = np.arange(len(order))

    # (a) partial correlation controlling the band energy -- the degeneracy test
    cols = {"const": "#7f7f7f", "AE": "#1f77b4", "PC": "#2ca02c"}
    h = 0.26
    for i, sch in enumerate(SCHEMES):
        vals = []
        for k in order:
            r = sub[(sub.scheme == sch) & (sub.metric == k)]["p_P_given_E"].mean()
            vals.append(0.0 if not np.isfinite(r) else r)
        ax[0].barh(y + (i - 1) * h, vals, height=h, color=cols[sch],
                   label=r"$\Sigma_p$: %s" % sch)
    ax[0].axvline(0, color="k", lw=0.9)
    ax[0].set_yticks(y, order)
    ax[0].set_xlabel(r"partial Spearman $\rho$(metric, $\log P_{Pc5}$ $\vert$ $\log E_{Pc5}$)")
    ax[0].set_title("(a) Does the dissipation rate carry information\n"
                    "beyond the band energy?")
    # The legend and the "const: rho == 0" annotation both sat in the lower-right
    # corner and overprinted each other (caught by a figure review), and after
    # moving it out the legend then landed on the x label.  Both legends are now
    # anchored well below the axes, in a band the figure reserves for them.
    ax[0].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                 ncol=3, frameon=False)
    ax[0].grid(alpha=0.22, axis="x", lw=0.4)
    ax[0].set_xlim(-0.25, 0.45)
    # this note sat in the lower-right of the axes and overprinted the end of a bar;
    # it now sits under the legend, where nothing else is drawn
    ax[0].text(0.5, -0.46,
               r"$const$: $\rho\equiv0$ by construction ($P\propto E$), "
               "which is why that series has no visible bar",
               transform=ax[0].transAxes, ha="center", va="top",
               fontsize=9.5, color="#555555")

    # (b) with an AE-modulated Sigma_p, rho(.,logE) and rho(.,logP) separate
    ae = sub[sub.scheme == "AE"].groupby("metric")[["rho_E", "rho_P"]].mean()
    ae = ae.reindex(order)
    for i, k in enumerate(order):
        ax[1].plot([ae.loc[k, "rho_E"], ae.loc[k, "rho_P"]], [i, i],
                   "-", color="#bbbbbb", lw=1.6, zorder=1)
    ax[1].plot(ae["rho_E"], y, "o", ms=8, color="#7f7f7f",
               label=r"vs $\log E_{Pc5}$ (energy)")
    ax[1].plot(ae["rho_P"], y, "s", ms=8, color="#d62728",
               label=r"vs $\log P_{Pc5}$ (dissipation)")
    ax[1].axvline(0, color="k", lw=0.9)
    ax[1].set_yticks(y, order)
    ax[1].set_xlabel(r"Spearman $\rho$")
    ax[1].set_title("(b) With an AE-modulated $\\Sigma_p$ the energy\n"
                    "and dissipation axes separate")
    ax[1].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                ncol=2, frameon=False)
    ax[1].margins(x=0.14)   # the outermost point was flush against the frame
    ax[1].grid(alpha=0.22, axis="x", lw=0.4)

    # Title removed: the figure caption in the manuscript carries the description.
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, wspace=0.30)
    fig.savefig(RES / "e2_sigma.png", dpi=300, bbox_inches="tight")
    print("[save] e2_sigma.png", flush=True)


if __name__ == "__main__":
    main()
