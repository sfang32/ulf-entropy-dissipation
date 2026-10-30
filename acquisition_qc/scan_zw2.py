#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scan Meridian Phase-II FGMM (zw2) from the B015 corpus link.

Data layout (B015/dataset/data/zw2_fgmm -> /data/ZW/file/success):
  {ROOT}/{STATION}/FGMM01/PGCV/{YYYY}/{YYYYMM}/{YYYYMMDD}/{YYYYMMDDHH}/0/
      {STATION}_FGMM01_PGCV_L1_DAY_{YYYYMMDD}000000_V01.00.txt

One file = one full day, 86400 records @ 1 Hz, columns:
  HH:MM:SS  DoY  Deta_H  Deta_Z  Deta_D  Temp
Header (#) lines carry the station geographic coordinates directly.

Outputs a station catalogue CSV: station, glat, glon, alt, years, n_days,
plus the list of available years for each station.
"""
import glob
import os
import re
import sys

ROOT = ("/data/ssd-stor/YLK/YLK-BUFFER/NSSC-FSF/B015/"
        "dataset/data/zw2_fgmm")

STA_RE = re.compile(r"#Station:\s*([A-Z]+)\(\s*([-\d.]+)E,\s*([-\d.]+)N"
                    r"(?:,\s*([-\d.]+)m)?\)")


def main(root=ROOT, out="zw2_stations.csv"):
    stations = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
    rows = []
    for st in stations:
        pat = os.path.join(root, st, "FGMM01", "PGCV", "*", "*", "*", "*",
                           "0", "%s_FGMM01_PGCV_L1_DAY_*.txt" % st)
        files = sorted(glob.glob(pat))
        if not files:
            rows.append((st, "", "", "", "", 0, ""))
            print("%-6s  NO_DATA" % st)
            continue
        years = sorted({f.split(os.sep)[-6] for f in files})
        # station coordinates from the header of the first file
        glon = glat = alt = ""
        with open(files[0], encoding="utf-8-sig", errors="replace") as fh:
            for _ in range(6):
                line = fh.readline()
                m = STA_RE.search(line)
                if m:
                    glon, glat, alt = m.group(2), m.group(3), m.group(4) or ""
                    break
        rows.append((st, glat, glon, alt, len(years), len(files),
                     ";".join(years)))
        print("%-6s  glat=%-7s glon=%-8s alt=%-7s days=%5d years=%s"
              % (st, glat, glon, alt, len(files), ",".join(years)))

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("station,glat,glon,alt,n_years,n_days,years\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print("\n[wrote] %s  (%d stations)" % (out, len(rows)))


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else "zw2_stations.csv")
