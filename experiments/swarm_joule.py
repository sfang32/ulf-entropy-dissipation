#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""swarm_joule.py -- Swarm MAG -> field-aligned current (FAC) and Joule heating rate.

v8 E12a prototype: give the entropy-dissipation claim an INDEPENDENT truth value
(magnetosphere-ionosphere Joule heating), instead of the empirical Sigma_P placeholder.

Method (single-satellite along-track gradient; Luehr 1996, Ritter et al. 2013):
  B      : Swarm MAGx_LR/CA 1 Hz, NEC components (North-East-Centre) [nT]
  s      : along-track distance from Lat/Lon/Radius [m]
  dB     : 500-km high-pass of B -> isolates the FAC-scale perturbation
  j_par  = (1/mu0) * d(dB_east)/ds                       [A/m^2]
           (orbit is near-polar, so the horizontal cross-track direction is ~east)
  Q_J    = j_par^2 / Sigma_P
           practical units: Q_J[mW/m^2] = j_par[muA/m^2]^2 / Sigma_P[S]
           (Ritter et al. 2013 convention; assume height-integrated Sigma_P)

Sanity targets:  quiet   j_par ~0.1-1  uA/m^2, Q_J ~0.1-1   mW/m^2
                 active  j_par ~1-10   uA/m^2, Q_J ~1-100   mW/m^2

Usage (server, anaconda3):
  python swarm_joule.py --sat A --date 2015-03-17 --sigma 5 --outdir results
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import cdflib
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU0 = 4.0 * np.pi * 1e-7
RE = 6371.0e3
SW_ROOT = Path("/data/ftp/fsf/databases/opensource/SWARM")


def load_swarm(sat, date, root=SW_ROOT):
    """Read one day of Swarm MAG (CA/MDR product)."""
    ymd = date.strftime("%Y%m%d")
    pats = [str(root / f"Sat_{sat}" / f"*MAG{sat}_*{ymd}*MDR_MAG*.cdf"),
            str(root / f"Sat_{sat}" / f"*MAG{sat}_*{ymd}*.cdf")]
    files = []
    for p in pats:
        files = sorted(glob.glob(p))
        if files:
            break
    if not files:
        raise SystemExit("no Swarm file for %s %s (%s)" % (sat, ymd, pats[0]))

    c = cdflib.CDF(files[0])
    t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Timestamp")))
    b = c.varget("B")
    d = pd.DataFrame({
        "time": t,
        "lat": c.varget("Latitude"),
        "lon": c.varget("Longitude"),
        "radius": c.varget("Radius"),
        "B_N": b[:, 0], "B_E": b[:, 1], "B_C": b[:, 2],
    })
    d["alt"] = d["radius"] - RE
    d = d.dropna(subset=["lat", "lon", "B_E"]).reset_index(drop=True)
    print("[sw] %s  %s  n=%d  %s .. %s" % (sat, Path(files[0]).name, len(d),
                                           d.time.iloc[0], d.time.iloc[-1]))
    return d


def along_track_m(d):
    """Cumulative along-track distance [m] (great-circle + radial component)."""
    lat = np.radians(d["lat"].to_numpy())
    lon = np.radians(d["lon"].to_numpy())
    r = d["radius"].to_numpy()
    dlat = np.diff(lat, prepend=lat[0])
    dlon = np.diff(lon, prepend=lon[0])
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi          # wrap dateline
    dx = r * dlat
    dy = r * np.cos(lat) * dlon
    dr = np.diff(r, prepend=r[0])
    step = np.sqrt(dx ** 2 + dy ** 2 + dr ** 2)
    return np.cumsum(step)


def _despike(x, s, max_grad_nT_per_km=50.0):
    """Remove single-sample glitches BEFORE filtering.

    Swarm CA data contains isolated ~1000-nT jumps (attitude/instrument glitches,
    densest near the poles). If left in, the high-pass turns them into fake FAC
    spikes of tens of uA/m^2. Real FAC gradients are ~1-10 nT/km, so a 50 nT/km
    threshold removes glitches without touching the physical signal.
    """
    g = np.abs(np.gradient(x, s)) * 1e3                 # nT/m -> nT/km
    bad = ~np.isfinite(g) | (g > max_grad_nT_per_km)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return x, 0
    y = x.copy()
    y[bad] = np.nan
    idx = np.arange(y.size)
    good = np.isfinite(y)
    if good.sum() > 2:
        y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y, n_bad


def fac_and_joule(d, sigma_p, hp_km=500.0, fs_nominal=1.0, smooth=7,
                  gap_km=50.0):
    """Return (j_par [uA/m^2], Q_J [mW/m^2], fc, s, n_glitch)."""
    s = along_track_m(d)
    be, n_glitch = _despike(d["B_E"].to_numpy(), s)

    v = np.gradient(s) * fs_nominal                       # ~m per sample
    lam = np.median(v[np.isfinite(v) & (v > 0)])
    fc = (lam * 1e-3 / hp_km) * fs_nominal if lam > 0 else 0.01
    fc = float(np.clip(fc, 1e-4, 0.45))
    sos = sps.butter(2, fc, btype="high", fs=fs_nominal, output="sos")
    dbe = sps.sosfiltfilt(sos, be - np.nanmean(be))
    if smooth > 1:                                        # extra de-spiking of the
        dbe = sps.medfilt(dbe, kernel_size=smooth)        # filtered perturbation
    dj = np.gradient(dbe, s)                              # nT/m
    jpar = (dj * 1e-9) / MU0 * 1e6                        # uA/m^2

    # do not differentiate across data gaps
    for k in np.where(np.diff(s) > gap_km * 1e3)[0]:
        jpar[max(0, k - 1):k + 3] = np.nan
    qj = (jpar ** 2) / sigma_p                            # mW/m^2 (Ritter convention)
    return jpar, qj, fc, s, n_glitch, dbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", default="A")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--sigma", type=float, default=5.0,
                    help="height-integrated Pedersen conductance [S]")
    ap.add_argument("--hp-km", type=float, default=500.0)
    ap.add_argument("--outdir",
                    default="/data/ssd-stor/fsf/workspaces/research_ulf/"
                            "ground_pc5/v8_entropy_dissipation/results")
    a = ap.parse_args()

    date = pd.Timestamp(a.date)
    d = load_swarm(a.sat, date)
    jpar, qj, fc, s, n_glitch, dbe = fac_and_joule(d, a.sigma, a.hp_km)
    d["s_km"] = s * 1e-3
    d["dB_E_filt"] = dbe
    d["jpar_uAm2"] = jpar
    d["QJ_mWm2"] = qj

    # --- extreme-value audit (are the largest FAC values physical or artefacts?) ---
    kk = int(np.nanargmax(np.abs(jpar)))
    ds_all = np.diff(s)
    near_gap = bool(np.any(ds_all[max(0, kk - 3):kk + 3] > 50e3))
    print("[sw] max |j_par| = %.2f uA/m^2 at %s  lat=%.1f lon=%.1f  "
          "dB_filt=%.1f nT  near_gap=%s"
          % (abs(jpar[kk]), d["time"].iloc[kk], d["lat"].iloc[kk],
             d["lon"].iloc[kk], dbe[kk], near_gap))
    top = np.argsort(-np.abs(np.nan_to_num(jpar)))[:5]
    for k in top:
        print("[sw]   top: %s lat=%.1f |j|=%.2f QJ=%.2f"
              % (d["time"].iloc[k], d["lat"].iloc[k], abs(jpar[k]), qj[k]))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "swarm_joule_%s_%s" % (a.sat, date.strftime("%Y%m%d"))
    csv = out / (tag + ".csv")
    d.to_csv(csv, index=False)
    print("[sw] wrote %s (%d pts, fc=%.4f Hz, glitches removed=%d)"
          % (csv, len(d), fc, n_glitch))

    def _stat(mask, lab):
        j, q = np.abs(jpar[mask]), qj[mask]
        j, q = j[np.isfinite(j)], q[np.isfinite(q)]
        if j.size == 0:
            print("[sw] %-18s (empty)" % lab)
            return
        print("[sw] %-18s n=%6d  |j| med=%.2f p95=%.2f p99=%.2f uA/m^2  |  "
              "Q_J med=%.3f p95=%.2f p99=%.2f mW/m^2"
              % (lab, j.size, np.median(j), np.percentile(j, 95),
                 np.percentile(j, 99), np.median(q), np.percentile(q, 95),
                 np.percentile(q, 99)))

    lat = d["lat"].to_numpy()
    _stat(np.abs(lat) > 55, "|lat|>55 (auroral)")
    _stat((np.abs(lat) > 30) & (np.abs(lat) <= 55), "30<|lat|<=55")
    _stat(np.abs(lat) <= 30, "|lat|<=30")
    _stat(np.isfinite(jpar), "all")

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 8.4), sharex=True)
    ax[0].plot(d["s_km"], d["B_E"], color="#c8c8c8", lw=0.5,
               label=r"raw $B_E$ (main field, for reference)")
    ax[0].plot(d["s_km"], d["dB_E_filt"], color="#1f77b4", lw=0.9,
               label=r"$\Delta B_E$ (%.0f-km high-pass)" % a.hp_km)
    ax[0].axhline(0, color="gray", lw=0.6, ls="--")
    ax[0].set_ylabel(r"$B_E$ / $\Delta B_E$ [nT]", labelpad=8)
    ax[0].legend(fontsize=9, loc="upper right", ncol=1, framealpha=0.9)
    ax[0].set_title("Swarm-%s  %s   FAC / Joule heating (1 Hz MAG, %.0f-km "
                    "high-pass, %d-pt despike, $\\Sigma_P$=%.0f S)"
                    % (a.sat, a.date, a.hp_km, 7, a.sigma), fontsize=11)
    ax[1].plot(d["s_km"], jpar, color="#d62728", lw=0.7)
    ax[1].axhline(0, color="gray", lw=0.6, ls="--")
    ax[1].set_ylabel(r"$j_\parallel$ [$\mu$A/m$^2$]", labelpad=8)
    ax[2].plot(d["s_km"], qj, color="#2ca02c", lw=0.7)
    ax[2].set_ylabel(r"$Q_J$ [mW/m$^2$]", labelpad=8)
    ax[2].set_xlabel("along-track distance [km]")
    for x in ax:
        x.grid(alpha=0.25, lw=0.4)
        x.set_xlim(d["s_km"].min(), d["s_km"].max())
    hi_idx = np.where(np.abs(lat) > 55)[0]
    if hi_idx.size:
        segs = np.split(hi_idx, np.where(np.diff(hi_idx) > 1)[0] + 1)
        for x in ax:
            for sg in segs:                       # one band per polar crossing
                x.axvspan(d["s_km"].iloc[sg[0]], d["s_km"].iloc[sg[-1]],
                          color="orange", alpha=0.12, lw=0)
            x.text(0.5, 0.985, "shaded = auroral band |lat|>55",
                   transform=x.transAxes, ha="center", va="top",
                   fontsize=9, color="#a05000")
    fig.tight_layout()
    png = out / (tag + ".png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print("[sw] figure -> %s" % png)


if __name__ == "__main__":
    main()
