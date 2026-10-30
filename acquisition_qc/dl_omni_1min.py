#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dl_omni_1min.py -- download OMNI HRO 1-min monthly CDFs for the v8 years.

Source (verified reachable 2026-10-02): NASA SPDF
    https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro_1min/<year>/
    omni_hro_1min_<YYYY><MM>01_v01.cdf      (monthly; fallback omni_hro2_1min_...)

Needed because the v8 Phase-II replication (2025/2026) requires an auroral-activity
driver for the conductance-modulated degeneracy test (§3.2), and the mirror on the
cluster only carries 2010-2022.

Usage:
  python dl_omni_1min.py --years 2025 2026 --out /tmp/omni_1min
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request

BASE = "https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro_1min"
UA = {"User-Agent": "v8_ulf/1.0"}


def month_names(year):
    """Monthly file names that actually exist in the year directory."""
    url = "%s/%d/" % (BASE, year)
    try:
        html = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=60
        ).read().decode("utf-8", "ignore")
    except Exception as e:
        print("  [list] %d FAILED: %s: %s" % (year, type(e).__name__, e))
        return []
    found = sorted(set(re.findall(r"omni_hro\d?_1min_\d{8}_v\d+\.cdf", html)))
    # prefer the hro_ (not hro2_) product when both exist for a month
    by_month = {}
    for f in found:
        mm = f.split("_")[3][4:6]
        if mm not in by_month or "hro2" in by_month[mm]:
            by_month[mm] = f
    return [by_month[k] for k in sorted(by_month)]


def fetch(year, name, outdir, timeout=300):
    url = "%s/%d/%s" % (BASE, year, name)
    dest = os.path.join(outdir, str(year), name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        print("  = %s (already present, %.1f MB)"
              % (name, os.path.getsize(dest) / 1e6))
        return dest
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            data = r.read()
        if len(data) < 100_000:
            raise ValueError("suspiciously small (%d bytes)" % len(data))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        print("  + %s  %.1f MB" % (name, len(data) / 1e6))
        return dest
    except Exception as e:
        print("  ! %s FAILED: %s: %s" % (name, type(e).__name__, e))
        if os.path.isfile(dest):
            os.remove(dest)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--out", default="/tmp/omni_1min")
    a = ap.parse_args()

    ok = bad = 0
    for y in a.years:
        names = month_names(y)
        print("[%d] %d monthly files listed" % (y, len(names)))
        for nm in names:
            if fetch(y, nm, a.out):
                ok += 1
            else:
                bad += 1
    print("\n[done] %d downloaded/verified, %d failed  -> %s" % (ok, bad, a.out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
