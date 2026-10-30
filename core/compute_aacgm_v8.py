"""compute_aacgm_v8.py -- true AACGM-v2 magnetic latitude for v8 stations.

Reuses the exact convention of project_ts/server/compute_aacgm.py
(aacgmv2 2.7 API: aacgmv2.convert_latlon(lat, lon, height_km, dtime)),
but with a 2015 reference epoch (v8 data year) instead of 2017-09-15.

Run on the server (torch env has aacgmv2 2.7.1):
  /data/ssd-stor/fsf/anaconda3/envs/torch/bin/python compute_aacgm_v8.py IN.csv OUT.csv [YYYY-MM-DD HH:MM]
Input csv needs glat/glon (or lat/lon, or Glat/glong) columns.
"""
import csv, sys, datetime

import aacgmv2

DEFAULT_REF = datetime.datetime(2015, 7, 1, 0, 0)   # v8 data year, mid-year reference
PAIRS = (("glat", "glon"), ("lat", "lon"), ("Glat", "glong"))


def pick(r):
    for a, b in PAIRS:
        if a in r and b in r:
            try:
                return float(r[a]), float(r[b])
            except (TypeError, ValueError):
                pass
    raise KeyError("no lat/lon pair in %s" % list(r))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    ref = (datetime.datetime.strptime(sys.argv[3], "%Y-%m-%d %H:%M")
           if len(sys.argv) > 3 else DEFAULT_REF)

    rows = list(csv.DictReader(open(src)))
    res = []
    for r in rows:
        try:
            glat, glon = pick(r)
        except (KeyError, ValueError):
            # stations whose header carries no coordinates (a few Phase-II sites)
            res.append(dict(r, aacgm_lat="", aacgm_lon="", abs_aacgm="",
                            aacgm_mlt=""))
            continue
        try:
            mlat, mlon, mlt = aacgmv2.convert_latlon(glat, glon, 0, ref)
        except Exception:
            mlat = mlon = mlt = float("nan")
        r2 = dict(r)
        r2["aacgm_lat"] = round(mlat, 3)
        r2["aacgm_lon"] = round(mlon, 3)
        r2["abs_aacgm"] = round(abs(mlat), 2)
        r2["aacgm_mlt"] = round(mlt, 3)
        res.append(r2)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print("[save] %s  n=%d  ref=%s" % (out, len(res), ref))

    zw = {"MHT", "MZL", "NAT", "SSL", "MLS", "JFT", "PXT", "LAT", "SYS", "SYT", "HZT"}
    for r in res:
        st = r.get("station", r.get("code", ""))
        if st in zw or r.get("aacgm_lat") not in ("", None):
            g = pick(r)
            print("  %-6s geo=(%8.3f,%9.3f)  AACGM=%7.2f  lon=%9.2f"
                  % (st, g[0], g[1], r["aacgm_lat"], r["aacgm_lon"]))
    print("AACGM_V8_DONE")


if __name__ == "__main__":
    main()
