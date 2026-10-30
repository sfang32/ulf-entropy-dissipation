#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""swarm_timeseries.py -- aggregate Swarm polar crossings into a Q_J(t) series.

v8 E12: turn the per-swath FAC profile into a TIME SERIES of ionospheric
dissipation, to be compared with ground entropy and storm indices.

One point per auroral-zone crossing (|lat| > LAT_MIN), per satellite:
  t_mid, n_pts, lat range, |j_par| stats, j_par^2 mean, Sigma_P, Q_J mean

Sigma_P modes:
  const : fixed conductance (baseline)
  ae    : Robinson-type  Sigma_P = 2.0 + 0.88*sqrt(AE)  [S]   (Moen & Brekke 1993)

Usage (server):
  python swarm_timeseries.py --sats A B --start 2015-01-01 --end 2015-12-31 \
      --sigma-mode ae
"""
from __future__ import annotations

import argparse
import glob
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import cdflib

import sys
sys.path.insert(0, str(Path(__file__).parent))
from swarm_joule import load_swarm, fac_and_joule   # reuse loader + physics

OMNI = Path("/data/ftp/fsf/databases/data_omni/omni_1min")


def load_omni_ae(year):
    """AE index series for a year (1-min, OMNI high-res)."""
    frames = []
    for f in sorted(glob.glob(str(OMNI / str(year) / ("omni_hro_1min_%d*.cdf" % year)))):
        c = cdflib.CDF(f)
        t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
        x = np.asarray(c.varget("AE_INDEX"), dtype=float)
        frames.append(pd.Series(np.where(np.abs(x) > 1e30, np.nan, x), index=t))
    if not frames:
        return None
    s = pd.concat(frames).sort_index()
    return s[~s.index.duplicated()]


def sigma_p_from_ae(ae):
    """Height-integrated Pedersen conductance [S] from AE [nT]."""
    return 2.0 + 0.88 * np.sqrt(np.clip(ae, 0.0, None))


def crossings(d, jpar, lat_min=55.0, min_pts=20):
    """Split one day into auroral-zone crossings and summarise each."""
    lat = d["lat"].to_numpy()
    m = np.abs(lat) > lat_min
    idx = np.where(m)[0]
    if idx.size == 0:
        return []
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    rows = []
    for sg in segs:
        if sg.size < min_pts:
            continue
        j = np.abs(jpar[sg])
        j = j[np.isfinite(j)]
        if j.size < min_pts:
            continue
        q = j ** 2                                    # uA^2/m^4 (Q_J = q / Sigma_P)
        rows.append({
            "t_mid": d["time"].iloc[sg[len(sg) // 2]],
            "n_pts": int(j.size),
            "lat_absmin": float(np.min(np.abs(lat[sg]))),
            "lat_absmax": float(np.max(np.abs(lat[sg]))),
            "jpar_abs_mean": float(np.mean(j)),
            "jpar_abs_med": float(np.median(j)),
            "jpar_abs_p95": float(np.percentile(j, 95)),
            "jpar_sq_mean": float(np.mean(q)),
            "jpar_sq_p95": float(np.percentile(q, 95)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sats", nargs="+", default=["A", "B"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sigma-mode", choices=["const", "ae"], default="ae")
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--lat-min", type=float, default=55.0)
    ap.add_argument("--outdir",
                    default="/data/ssd-stor/fsf/workspaces/research_ulf/"
                            "ground_pc5/v8_entropy_dissipation/results")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ae_ser = None
    if a.sigma_mode == "ae":
        ae_ser = load_omni_ae(pd.Timestamp(a.start).year)
        if ae_ser is None:
            print("[ts] WARNING: no OMNI AE -> falling back to const sigma")
            a.sigma_mode = "const"
        else:
            print("[ts] OMNI AE loaded: %d pts  %s .. %s"
                  % (len(ae_ser), ae_ser.index[0], ae_ser.index[-1]))

    all_rows = []
    day, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    while day <= end:
        for sat in a.sats:
            try:
                d = load_swarm(sat, day)
            except SystemExit as e:
                print("[ts] skip %s %s (%s)" % (sat, day.date(), e))
                continue
            jpar, qj, fc, s, ng, _dbe = fac_and_joule(d, a.sigma)
            rows = crossings(d, jpar, a.lat_min)
            for r in rows:
                r["sat"] = sat
                r["date"] = str(day.date())
                if ae_ser is not None:
                    ae = ae_ser.reindex([r["t_mid"]], method="nearest",
                                        tolerance="5min").iloc[0]
                    if np.isfinite(ae):
                        r["ae"] = float(ae)
                        r["sigma_p"] = float(sigma_p_from_ae(ae))
                    else:
                        r["ae"] = np.nan
                        r["sigma_p"] = a.sigma
                else:
                    r["ae"] = np.nan
                    r["sigma_p"] = a.sigma
                r["QJ_mean"] = r["jpar_sq_mean"] / r["sigma_p"]
                r["QJ_p95"] = r["jpar_sq_p95"] / r["sigma_p"]
            all_rows.extend(rows)
            print("[ts] %s %s: %d crossings" % (sat, day.date(), len(rows)),
                  flush=True)
        day += timedelta(days=1)

    if not all_rows:
        raise SystemExit("[ts] no crossings")
    df = pd.DataFrame(all_rows).sort_values("t_mid").reset_index(drop=True)
    csv = out / ("swarm_crossings_%s_%s.csv"
                 % (a.start.replace("-", ""), a.end.replace("-", "")))
    df.to_csv(csv, index=False)
    print("[ts] wrote %s (%d crossings)" % (csv, len(df)))
    g = df.groupby("sat")[["jpar_abs_mean", "QJ_mean", "sigma_p"]].median().round(3)
    print(g.to_string())
    print("[ts] all: QJ_mean median=%.3f p95=%.3f mW/m^2 | sigma_p median=%.2f S"
          % (df["QJ_mean"].median(), df["QJ_mean"].quantile(0.95),
             df["sigma_p"].median()))


if __name__ == "__main__":
    main()
