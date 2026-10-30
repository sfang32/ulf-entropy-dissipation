"""scan_carisma_layout.py -- audit the two CARISMA F01 body layouts.

The .F01.gz body is written in two layouts inside the same station-year:
  spaced       `20150101000000 18967.044 -2203.800 51180.891 .`
  run-together `20150509000000 -1143.350-46135.872-30139.569 .`
A whitespace split silently turns the second layout into all-NaN, so this script
counts how many days of each kind exist across the whole archive, using the same
regex the loader now uses (run_multiband._read_carisma_day) as the reference.

Run on the server:
  /data/ssd-stor/fsf/anaconda3/bin/python scan_carisma_layout.py
"""
from __future__ import annotations

import glob
import gzip
import os
import re
import sys
from collections import Counter

CARISMA = "/data/ftp/fsf/databases/data_yuliaoku/CARISMA/FGM/1Hz"
TS = re.compile(rb"^(\d{14})")
NUM = re.compile(rb"[-+]?\d+\.\d+")


def classify(path):
    """Return 'spaced' | 'runtogether' | 'empty' | 'bad' for one day file."""
    with gzip.open(path, "rb") as fh:
        fh.readline()
        n_spaced = n_run = n_other = 0
        for line in fh:
            if not TS.match(line):
                continue
            toks = line.split()
            if len(toks) >= 5:
                n_spaced += 1
            elif len(NUM.findall(line, 14)) >= 3:
                n_run += 1
            else:
                n_other += 1
        n = n_spaced + n_run + n_other
        if n == 0:
            return "empty", 0, 0, 0
        if n_other > 0.01 * n:
            return "bad", n_spaced, n_run, n_other
        return ("spaced" if n_spaced >= n_run else "runtogether"), n_spaced, n_run, n_other


def main():
    years = sys.argv[1:] or ["2015"]
    grand = Counter()
    print("%-6s %-6s %8s %8s %8s  %s" %
          ("ST", "YEAR", "spaced", "run-tog", "other", "verdict"))
    for st in sorted(os.listdir(CARISMA)):
        for y in years:
            files = sorted(glob.glob(os.path.join(CARISMA, st, y, "*", "*.F01.gz")))
            if not files:
                continue
            kinds, sp, rn, ot = Counter(), 0, 0, 0
            for f in files:
                k, a, b, c = classify(f)
                kinds[k] += 1
                sp += a
                rn += b
                ot += c
            grand.update(kinds)
            verdict = "OK" if kinds["bad"] == 0 and kinds["empty"] == 0 else "CHECK"
            flag = "  <== run-together present" if kinds["runtogether"] else ""
            print("%-6s %-6s %8d %8d %8d  %s%s" %
                  (st.upper(), y, kinds["spaced"], kinds["runtogether"],
                   kinds["bad"] + kinds["empty"], verdict, flag))
    print("\n=== totals over %s ===" % "+".join(years))
    for k, v in sorted(grand.items()):
        print("  %-14s %d" % (k, v))


if __name__ == "__main__":
    main()
