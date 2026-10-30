"""centered-dipole geomagnetic latitude: zero-dependency tool.

Mirrors the CGM convention used by v6 station table (calibration check included).
IGRF-12 2015.0 geomagnetic (dipole) north pole: 80.37N, 72.62W
Usage: python cgm_approx.py <coords_csv> [out_csv]
"""
import csv, math, sys, os

POLE_LAT, POLE_LON = 80.37, -72.62          # IGRF-12 2015.0 centered dipole north pole

def dip_lat(glat, glon):
    """magnetic (dipole) latitude in deg; glon east-positive."""
    p, pl = math.radians(glat), math.radians(POLE_LAT)
    dl = math.radians(glon - POLE_LON)
    ct = math.sin(p) * math.sin(pl) + math.cos(p) * math.cos(pl) * math.cos(dl)
    ct = max(-1.0, min(1.0, ct))
    theta = math.acos(ct)                    # magnetic colatitude
    return 90.0 - math.degrees(theta), theta

def lshell(theta):
    s = math.sin(theta)
    return float("inf") if s <= 1e-9 else 1.0 / (s * s)

def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_cgm.csv"
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            try:
                glat, glon = float(r["glat"]), float(r["glon"])
            except (KeyError, ValueError):
                continue
            ml, th = dip_lat(glat, glon)
            r["mlat_dipole"] = round(ml, 2)
            r["L"] = round(lshell(th), 3)
            rows.append(r)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote %d rows -> %s" % (len(rows), out))

    # --- calibration check: v6 8-station dipole latitudes ---
    chk = {"MHT": 43.9, "MZL": 39.8, "NAT": 34.6, "MLS": 25.0, "PXT": 22.0, "JFT": 21.0, "LAT": 19.6, "SYS": 9.7}
    lut = {r["code"]: r for r in rows}
    print("\n[calib] station  v6_CGM  dipole_calc  delta")
    for k, v in chk.items():
        r = lut.get(k)
        if r:
            print("  %-4s  %6.1f  %10.2f  %+6.2f" % (k, v, r["mlat_dipole"], r["mlat_dipole"] - v))
    b = lut.get("BMT")
    if b:
        print("\n[SSL] BMT(Beijing Ming Tombs) glat=%s glon=%s -> dip_lat=%.2f L=%.3f" % (b["glat"], b["glon"], b["mlat_dipole"], b["L"]))

    # --- E5b candidates: dipole lat 55-70 ---
    cand = sorted([r for r in rows if 55.0 <= r["mlat_dipole"] <= 70.0], key=lambda r: -r["mlat_dipole"])
    print("\n[E5b] dipole lat 55-70 deg candidates: %d stations" % len(cand))
    print("  code   glat    glon    dip_lat   L")
    for r in cand:
        print("  %-5s %6.2f %8.2f  %7.2f  %5.2f" % (r["code"], float(r["glat"]), float(r["glon"]), r["mlat_dipole"], r["L"]))

if __name__ == "__main__":
    main()
