"""pull_e5b_2015.py -- pull E5b high-latitude stations, 2015 full year, 1-min, week by week.

Stations (AACGM-v2 / 2015 ref):  KIR 64.80 (Sweden, IMAGE)  YKC 68.84 (Canada)
                                 BRW 70.31 (Alaska, polar-cap edge)
Replaces the earlier SOD/C03/BRW trio: C03 has NO 2015 data; SOD & BRW missed the summer
probe day (availability must be measured over the full year, not sampled).

Output: data/supermag_e5b/{ST}_2015_1min.csv.gz  (tval,time_utc,N,E,Z,mlt,glat,glon,decl,sza)
        data/supermag_e5b/e5b_coverage.csv      (per station: valid days, N-missing)
"""
import concurrent.futures
import os
import ssl
import sys
import urllib.request
from datetime import date, timedelta

_orig = urllib.request.urlopen
def _patched(url, *a, **k):
    k.pop("cafile", None)
    ctx = ssl.create_default_context(); ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return _orig(url, context=ctx, *a, **k)
urllib.request.urlopen = _patched

from supermag_api import supermag_api as sm
import numpy as np
import pandas as pd

# SuperMAG requires a registered user name as the "logon" argument.  It is an
# account identifier, not a credential, but it still names a person, so it is
# read from the environment here instead of being hard-coded in the deposit.
#   export SUPERMAG_LOGON=<your SuperMAG user name>
LOGON = os.environ.get("SUPERMAG_LOGON", "").strip()
if not LOGON:
    sys.exit("set SUPERMAG_LOGON to your registered SuperMAG user name "
             "(free registration at https://supermag.jhuapl.edu/)")

YEAR = 2015
STATIONS = ["KIR", "YKC", "BRW"]
# Output tree; override with SUPERMAG_E5B_OUT.  Default is relative to the CWD.
OUT = os.environ.get("SUPERMAG_E5B_OUT", os.path.join("data", "supermag_e5b"))
os.makedirs(OUT, exist_ok=True)


def flatten(df):
    """N/E/Z come back as dicts with 'nez'/'mean'/'ext'; keep nez only + time index."""
    t = pd.to_datetime(df["tval"], unit="s", utc=True)
    out = pd.DataFrame({"time_utc": t})
    for c in ("N", "E", "Z"):
        if c in df.columns:
            col = df[c]
            if isinstance(col.iloc[0], dict):
                out[c + "_nez"] = pd.to_numeric(
                    col.apply(lambda d: d.get("nez") if isinstance(d, dict) else None),
                    errors="coerce").values
            else:
                out[c + "_nez"] = pd.to_numeric(col, errors="coerce").values
    for c in ("mlt", "glat", "glon", "decl", "sza"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").values
    return out


def run_station(st):
    frames, d, end = [], date(YEAR, 1, 1), date(YEAR, 12, 31)
    while d <= end:
        ndays = min(7, (end - d).days + 1)
        try:
            ok, df = sm.SuperMAGGetData(LOGON, [d.year, d.month, d.day, 0, 0, 0],
                                        ndays * 86400, "all", st)
        except Exception:
            ok, df = False, None
        if ok and df is not None and len(df) > 0:
            frames.append(flatten(df))
        d += timedelta(days=ndays)
    if not frames:
        return st, None, "NO DATA AT ALL"
    full = pd.concat(frames).drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
    full = full.set_index("time_utc")
    out = os.path.join(OUT, "%s_%d_1min.csv.gz" % (st, YEAR))
    full.to_csv(out, compression="gzip")
    daily = full["N_nez"].resample("1D").apply(lambda s: s.notna().sum())
    good_days = int((daily >= 1200).sum())      # >=20h of 24h -> usable day
    partial = int(((daily > 0) & (daily < 1200)).sum())
    miss = int(full["N_nez"].isna().sum())
    return st, {"n_pts": len(full), "good_days": good_days, "partial_days": partial,
                "N_missing": miss, "glat": float(full["glat"].median()),
                "file": out}, "OK"


def main():
    rows = []
    print("=== E5b pull: %s %d (1-min, weekly chunks) ===" % (",".join(STATIONS), YEAR),
          flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for st, info, msg in ex.map(run_station, STATIONS):
            if info is None:
                print("[%s] %s" % (st, msg), flush=True)
                continue
            print("[%s] points=%d good_days=%d/365 partial=%d N_miss=%d glat=%.2f -> %s"
                  % (st, info["n_pts"], info["good_days"], info["partial_days"],
                     info["N_missing"], info["glat"], info["file"]), flush=True)
            info["station"] = st
            rows.append(info)
    if rows:
        t = pd.DataFrame(rows)[["station", "n_pts", "good_days", "partial_days",
                                "N_missing", "glat", "file"]]
        t.to_csv(os.path.join(OUT, "e5b_coverage.csv"), index=False)
        print(t.to_string(index=False), flush=True)
    print("E5B_PULL_DONE", flush=True)


if __name__ == "__main__":
    main()
