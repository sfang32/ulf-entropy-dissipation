"""run_multiband.py -- v8 multi-band entropy/dissipation pipeline (direction C).

Per 方老师 (2026-09-30): do NOT lock the analysis to Pc5. Run the SAME pipeline over
Pc3 / Pc4 / Pc5 / full-band so the entropy-vs-dissipation calibration can be compared
ACROSS channels -- "which band does entropy actually track?" is itself a result.

Per 10-min/1-h window (same window for all bands, raw series for entropy):
    entropy battery (raw window)   vs   E_b = mean(filt_b^2) [nT^2]
                                        P_b = mean(dB_b^2/(mu0^2 Sigma_p)) [W/m^2]
Bands: Pc3 23-100 mHz, Pc4 6.7-23 mHz, Pc5 1.6-6.7 mHz, "all" = raw demeaned (no filter).

Two data sources (both non-overlapping windows):
  ZW (Meridian 1 Hz):  --src zw --station MHT --year 2015 --win 600  --m-pe 4
  SM (SuperMAG 1 min): --src sm --station KIR --year 2015 --win 3600 --m-pe 3
                       (E5b: SuperMAG has no 1-sec product -> 1-h window, m=3)
  CA (CARISMA 1 Hz):   --src carisma --station RANK --year 2015 --win 600 --m-pe 4
                       (replaces SM: 1 Hz native, so the SAME 600 s / m=4 window
                       as ZW -- the high-latitude arm becomes like-for-like)

Output: <outdir>/{mb}_{src}_{station}{year}_multiband.csv  (+ scatter png)
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import entropy_battery as eb
import deposition_rate as dr

ZW = Path("/data/ssd-stor/fsf/databases/data_ulf/dataset_pc/ZW")
# Raw Meridian-I FGM "DMD L01 01D" product from the B015 corpus.  Same 1 Hz H/D/Z/T
# content as the processed ZW tree above, but it covers 2010-2024 whereas the
# processed product stops at 2022 -- so this is the only route to 2023 and 2024.
# Layout of one daily file (verified 2026-10-03, probe_zw_raw.py):
#   9 header tokens (YYYYMMDD, counter, station code, product codes, 4 channel ids)
#   followed by 86400 groups of 4 values: H, D, Z, T.  "NULL" marks a missing value.
#   So the name 'DMD' notwithstanding, this product is 1 Hz, one file per day.
ZW_RAW = Path("/data/ssd-stor/fsf/workspaces/project_yuliaoku/products/YLK/"
              "YLK-BUFFER/NSSC-FSF/B015/dataset/data/zw1_fgm_dat")
SM_DIR = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
              "v8_entropy_dissipation/data/supermag_e5b")
# Meridian Phase-II FGMM via the B015 corpus link (~ /data/ZW/file/success).
# PGCV L1 DAY product: one file = one full day, 86400 records @ 1 Hz, HDZ.
ZW2 = Path("/data/ssd-stor/YLK/YLK-BUFFER/NSSC-FSF/B015/"
           "dataset/data/zw2_fgmm")
# CARISMA (Canadian Array for Realtime Magnetospheric Observations) 1 Hz FGM.
# Layout: {root}/{sta lowercase}/{YYYY}/{MM}/{YYYYMMDD}{STA uppercase}.F01.gz
# One file = one full day = 86400 records @ 1 Hz.  Body lines are
#   YYYYMMDDHHMMSS  X  Y  Z  .
# with X/Y/Z in nT and Z along the LOCAL VERTICAL (verified numerically: at ANNA
# |Z| = 51181 nT matches the IGRF local-vertical Z, whereas a geomagnetic-dipole Z
# would be ~42784 nT; the header flag "GEODETIC" describes the station position).
# We therefore reduce to the horizontal intensity H = sqrt(X^2 + Y^2), which is
# invariant under any rotation of the horizontal plane and so does not depend on
# whether X/Y are true-north/east or BGGF-north/east.  That is the same scalar the
# ZW chain uses, so the two networks are directly comparable.
CARISMA = Path("/data/ftp/fsf/databases/data_yuliaoku/CARISMA/FGM/1Hz")

# AACGM-v2 magnetic latitude (aacgmv2 2.7, ref 2015-07-01). ZW stations + E5b trio.
MLAT = {"MHT": 48.36, "MZL": 44.54, "NAT": 38.44, "SSL": 35.04, "MLS": 29.09,
        "PXT": 25.47, "JFT": 24.85, "HZT": 24.35, "LAT": 23.66, "SYT": 21.41,
        "SYS": 13.16,
        # --- Meridian Phase-II FGMM (zw2; AACGM ref 2025-07-01) ---
        "OMOHE": 48.70, "OYAKS": 45.44, "OHUTB": 40.66, "OXLHT": 39.17,
        "OKSHI": 35.66, "OJIYG": 35.60, "OCSSL": 35.43, "OHLAN": 34.33,
        "ODLZS": 34.31, "OQIMO": 34.05, "OTYJY": 32.91, "OTANC": 29.47,
        "OGAER": 28.07, "OPIXI": 26.01, "OYICH": 25.78, "OENSH": 25.26,
        "OBATA": 25.09, "OHZXH": 24.73, "OLSCG": 24.72, "OSHYA": 21.90,
        "OLJGC": 21.80, "ONANA": 19.34, "OZHQI": 17.40, "OQIZH": 13.20,
        "OLEDO": 12.62,
        # E5b (SuperMAG, 1-min)
        "KIR": 64.80, "YKC": 68.84, "BRW": 70.31, "SOD": 64.11, "ABK": 65.39,
        "GIM": 65.44, "NAQ": 64.83, "FCC": 67.74,
        # --- CARISMA 1 Hz (aacgmv2 2.7.1, ref 2015-07-01, computed by
        #     compute_aacgm_v8.py from the geographic coordinates in each FGM
        #     file header; see data/carisma/stations_carisma_aacgm.csv) ---
        # All 25 sites sit in the NORTHERN hemisphere, so aacgm_lat > 0 throughout
        # and abs_aacgm == aacgm_lat.  The chain spans 52.19 (ANNA) to 77.76
        # (TALO); the 52-66 deg band is much denser than the old 3-station
        # SuperMAG arm, which is the point of the swap.
        "ANNA": 52.19, "OSAK": 55.28, "POLS": 54.34, "THRF": 57.28,
        "VULC": 57.25, "GULL": 57.76, "WGRY": 57.41, "WEYB": 58.07,
        "PINA": 59.41, "MSTK": 60.21, "LGRR": 61.23, "ISLL": 63.03,
        "OXFO": 64.02, "MCMU": 63.75, "GILL": 65.44, "DAWS": 65.77,
        "FCHP": 65.82, "FSMI": 66.89, "FSIM": 66.94, "RABB": 66.36,
        "BACK": 66.74, "FCHU": 67.74, "NORM": 69.34, "RANK": 71.64,
        "TALO": 77.76}

# tool_cor band edges (Hz); "all" handled separately (no filter)
WN = {"Pc3": (23e-3, 100e-3), "Pc4": (6.7e-3, 23e-3), "Pc5": (1.6e-3, 6.7e-3)}
BANDS = ["Pc3", "Pc4", "Pc5", "all"]
EDGE_SEC = 1800          # discard transient zone at day edges [s] (bandpass settle)
SCALAR_KEYS = ["SE_full", "SE_Pc3", "SE_Pc4", "SE_Pc5", "PE", "WPE",
               "SampEn", "Tsallis_q1.5", "CH_H", "CH_C", "SVD", "IR_ordinal"]


def band_filter(x, band, fs):
    """10th-order Butterworth sos bandpass, as tool_cor/core/tfa.py (zero-phase not used)."""
    if band == "all":
        return np.asarray(x, dtype=float) - np.nanmean(x)
    sos = sps.butter(N=10, Wn=WN[band], btype="bp", fs=fs, output="sos")
    return sps.sosfilt(sos, np.asarray(x, dtype=float) - np.nanmean(x))


def usable_bands(fs):
    """Bands that fit strictly inside Nyquist for this sampling rate.

    SuperMAG 1-min (fs=1/60 Hz) -> Nyquist 8.33 mHz: only Pc5 survives;
    Pc4 (6.7-23) and Pc3 (23-100) alias and MUST be skipped (same conclusion as e0c).
    """
    nyq = fs / 2.0
    ok = [b for b in ("Pc3", "Pc4", "Pc5") if WN[b][1] < nyq]
    skipped = [b for b in ("Pc3", "Pc4", "Pc5") if WN[b][1] >= nyq]
    if skipped:
        print("[mb] skip %s: above Nyquist (fs=%g Hz -> %.4f Hz)" % (skipped, fs, nyq),
              flush=True)
    return ok + ["all"]


def find_h_column(df):
    cols = {c.upper(): c for c in df.columns}
    for key in ("H", "DH", "HORIZONTAL"):
        if key in cols:
            return cols[key]
    for c in df.columns:
        cu = c.upper()
        if "H" in cu and "TIME" not in cu and "DATE" not in cu:
            return c
    raise KeyError(f"cannot find H column in {list(df.columns)}")


# ---------------------------------------------------------------- ZW (1 Hz)
def load_zw(station, year):
    base = ZW / station / str(year) / "10min" / "feature_1Hz"
    files = sorted(base.rglob("*.csv"))
    if not files:
        raise SystemExit(f"no feature_1Hz csv under {base}")
    days = OrderedDict()
    for f in files:
        try:
            t = pd.to_datetime(f.stem)
        except Exception:
            continue
        days.setdefault(t.date(), []).append((t, f))
    out = []
    for day, flist in days.items():
        ts, hs = [], []
        for t, f in sorted(flist, key=lambda x: x[0]):
            df = pd.read_csv(f)
            hs.append(df[find_h_column(df)].to_numpy(dtype=float))
            ts.append(t)
        if not hs:
            continue
        h_cat = np.concatenate(hs)
        t_cat = []
        for t, h in zip(ts, hs):
            t_cat.append(pd.date_range(start=t, periods=len(h), freq="s"))
        t_cat = t_cat[0].append(t_cat[1:]) if len(t_cat) > 1 else t_cat[0]
        s = pd.Series(h_cat, index=t_cat)
        s = s[~s.index.duplicated()]
        n_raw = s.notna().sum()
        s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="s"))
        s = s.interpolate(limit_direction="both")
        if n_raw < 0.5 * len(s):
            continue
        out.append((day, s))
    return out


# ------------------------------------------- Meridian-I raw daily files (1 Hz)
def load_zw_raw(station, year, root=ZW_RAW):
    """Read a year of Meridian-I FGM daily .dat files as (day, 1 Hz H series).

    Returns the same structure as load_zw() so that the two are interchangeable:
    a list of (date, pd.Series) at 1 Hz.  Values are nT; 'NULL' becomes NaN and is
    left NaN here, because the downstream band filter and entropy code already
    handle gaps, and interpolating before filtering would invent structure.
    """
    base = root / station / str(year)
    files = sorted(base.glob("*.dat"))
    if not files:
        raise SystemExit("no raw .dat under %s" % base)
    out = []
    for f in files:
        toks = f.read_text(errors="ignore").split()
        if len(toks) < 13:
            continue
        stamp = toks[0]
        try:
            date = pd.to_datetime(stamp, format="%Y%m%d")
        except ValueError:
            continue
        body = toks[9:]
        n = len(body) // 4
        if n < 3600:
            continue
        h = np.full(n, np.nan)
        for k in range(n):
            v = body[4 * k]
            if v != "NULL":
                try:
                    h[k] = float(v)
                except ValueError:
                    pass
        if np.isfinite(h).sum() < 0.5 * n:
            continue
        s = pd.Series(h, index=pd.date_range(date, periods=n, freq="s"))
        # Interpolate gaps exactly as load_zw() does.  This is not cosmetic: every
        # entropy measure in the battery is computed on the raw window, and a NaN
        # inside the window corrupts the ordinal patterns, which shifts every
        # entropy value (verified: without this line the band energy still agrees
        # with the processed product at corr = 1.000000 while CH_C agrees only at
        # 0.44).  Band energies are means and tolerate NaN; the entropies do not.
        s = s.interpolate(limit_direction="both")
        out.append((date.date(), s))
    out.sort(key=lambda x: x[0])
    return out


# ------------------------------------------- Meridian Phase-II FGMM (1 Hz)
def load_zw2(station, year, root=ZW2):
    """Meridian-II FGMM PGCV L1 DAY files: one file = one day @ 1 Hz.

    Layout: {root}/{st}/FGMM01/PGCV/{YYYY}/{YYYYMM}/{YYYYMMDD}/{YYYYMMDDHH}/0/
                {st}_FGMM01_PGCV_L1_DAY_{YYYYMMDD}000000_V01.00.txt
    Header (#) lines give the station geographic coordinates; the body is
    `HH:MM:SS DoY Deta_H Deta_Z Deta_D Temp` (missing = 999999.99).
    """
    base = Path(root) / station / "FGMM01" / "PGCV" / str(year)
    files = sorted(base.rglob(f"{station}_FGMM01_PGCV_L1_DAY_*.txt"))
    if not files:
        raise SystemExit(f"no zw2 FGMM day files under {base}")
    out = []
    for f in files:
        # date from the file name: ..._DAY_YYYYMMDD000000_V01.00.txt
        date_s = f.name.split("_DAY_")[1][:8]
        df = pd.read_csv(f, sep=r"\s+", comment="#", header=None,
                         names=["t", "doy", "H", "Z", "D", "T"],
                         encoding="utf-8-sig", engine="python",
                         na_values=["999999.99", "999999"])
        if len(df) and not str(df.iloc[0]["t"]).strip()[:1].isdigit():
            df = df.iloc[1:]          # drop the bare column-name row (Time DoY ...)
        idx = pd.to_datetime(date_s + " " + df["t"].astype(str),
                             format="%Y%m%d %H:%M:%S", errors="coerce")
        h = pd.to_numeric(df["H"], errors="coerce").to_numpy(dtype=float)
        s = pd.Series(h, index=idx)
        s = s[s.index.notna()]
        s = s[~s.index.duplicated()].sort_index()
        if s.empty or s.notna().sum() < 0.5 * 86400:
            continue
        s = s.reindex(pd.date_range(date_s + " 00:00:00", periods=86400,
                                    freq="s"))
        s = s.interpolate(limit_direction="both")
        out.append((pd.Timestamp(date_s).date(), s))
    return out


def zw2_header_coords(station, year, root=ZW2):
    """Geographic (lon, lat, alt) parsed from a Phase-II file header."""
    base = Path(root) / station / "FGMM01" / "PGCV" / str(year)
    for f in sorted(base.rglob(f"{station}_FGMM01_PGCV_L1_DAY_*.txt")):
        with open(f, encoding="utf-8-sig", errors="replace") as fh:
            for _ in range(6):
                line = fh.readline()
                if "#Station" in line:
                    return line.strip()
    return ""


# ------------------------------------------------------------ CARISMA (1 Hz)
_CA_TS = re.compile(rb"^(\d{14})")
_CA_NUM = re.compile(r"[-+]?\d+\.\d+")
# At the magnetic latitudes of this network (52-78 deg AACGM) the vertical
# component never drops below ~40000 nT, while X and Y stay well under 60000 nT.
# That ordering is what makes the glued-token split below unique instead of a
# guess; it is a physical constraint, not a tuned constant.
_CA_ABSZ_MIN = 40000.0
_CA_ABSXY_MAX = 60000.0
# 🔴 CARISMA's own missing-value fill, written into X and Z as 99999.999 with a
# trailing 'x' instead of the normal '.' (OXFO 2015-01..07, 209 of355 days; 306
# days network-wide in 2015).  It must be detected per ROW, not per file: a day
# can start filled and recover mid-day, and a file whose first line happens to be
# clean would otherwise pass a header-only check and contribute a fully
# interpolated series.  This is the same failure mode as the OMNI 999999.9 fill.
_CA_FILL = 99999.999


def _ca_split_glued(tok):
    """Split one whitespace token that holds two concatenated fixed-point numbers.

    CARISMA writes each component with %9.3f / %10.3f padding, so a value that
    exactly fills its field leaves no separator before the next one.  Splitting
    that token needs a rule; the |Z| > 40000 nT bound picks exactly one
    candidate.  Returns (a, b) or (None, None) when no split satisfies it.
    """
    if len(_CA_NUM.findall(tok)) >= 2:          # already two numbers, e.g. -1.2-3.4
        m = _CA_NUM.findall(tok)
        return m[0], m[1]
    for i in range(1, len(tok) - 4):
        a, b = tok[:i], tok[i:]
        if not re.fullmatch(r"[-+]?\d+\.\d{3}", a):
            continue
        if not re.fullmatch(r"[-+]?\d+\.\d{3}", b):
            continue
        fa, fb = float(a), float(b)
        if abs(fb) > _CA_ABSZ_MIN and abs(fa) < _CA_ABSXY_MAX:
            return a, b
    return None, None


def _read_carisma_day(path):
    """Parse one CARISMA .F01.gz day into a DataFrame [t, X, Y, Z].

    The body exists in three layouts that all occur within a single station-year,
    so any fixed-width or naive whitespace assumption silently corrupts data:

      spaced `20150101000000 15343.231  3726.594 53053.606 .`      (most days)
      glued  `20150101000000 -1143.350-46135.872-30139.569 .`      (WGRY 2015-05)
      glued  `20150101000000 14426.616   698.975200723.278 .`      (LGRR, OXFO)

    The third form is the dangerous one: the split happens when a value exactly
    fills its field width, so Y and Z end up in one token.  Reading that token as
    Y silently yields a wrong field (Z is then lost), and a regex that simply
    finds numbers concatenates Y+Z into a single bogus value.  Each line is
    therefore assembled from tokens, glued tokens are separated with the |Z| rule
    in _ca_split_glued, and every row is checked for |Z| > 40000 nT before it is
    accepted -- so an unparseable day is DROPPED rather than filled with garbage.
    """
    import gzip
    stamps, xs, ys, zs = [], [], [], []
    n_fill = n_drop = 0
    with gzip.open(path, "rb") as fh:
        fh.readline()                      # station/lat/lon/date/flags header
        for line in fh:
            m = _CA_TS.match(line)
            if not m:
                continue
            toks = line[14:].split()
            if toks and toks[-1] in (b".", b"x"):
                # 'x' marks a filled record, '.' a real one
                filled = toks[-1] == b"x"
                toks = toks[:-1]
            else:
                filled = False
            vals = []
            bad = filled
            for tk in toks:
                s = tk.decode("ascii", "replace")
                if not re.fullmatch(r"[-+]?\d+\.\d+", s):
                    a, b = _ca_split_glued(s)   # may still be 3 numbers
                    if a is None:
                        bad = True
                        break
                    vals.extend([a, b])
                else:
                    vals.append(s)
            if bad or len(vals) < 3:
                if filled:
                    n_fill += 1
                else:
                    n_drop += 1
                continue
            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            # Row-level fill test, independent of the trailing marker
            if (abs(x) > _CA_FILL * 0.9 or abs(z) > _CA_FILL * 0.9
                    or abs(y) > _CA_FILL * 0.9):
                n_fill += 1
                continue
            # Physical sanity gate: an unglued row must have a plausible vertical
            # component.  Catches both wrong splits and mis-read columns.
            if not (_CA_ABSZ_MIN < abs(z) < 90000.0):
                n_drop += 1
                continue
            stamps.append(m.group(1).decode())
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not stamps:
        return None
    out = pd.DataFrame({"t": stamps, "X": xs, "Y": ys, "Z": zs})
    if n_fill or n_drop:
        print("      [carisma] %s: %d filled + %d unparseable rows of%d usable"
              % (Path(path).name, n_fill, n_drop, len(stamps)), flush=True)
    return out


def load_carisma(station, year, root=CARISMA):
    """CARISMA 1 Hz FGM daily files -> [(day, 1 Hz H series)], same shape as load_zw().

    One file = one UTC day = 86400 records.  The header line is
        `STN  glat  glon  YYYYMMDD  GEODETIC  nT  1Hz`
    and the body is `YYYYMMDDHHMMSS  X  Y  Z  .`

    🔴 The body exists in FOUR variants that all occur inside the same
    station-year, and three of them corrupt the data silently, so the reader is a
    dedicated function (_read_carisma_day) rather than a pd.read_csv call:

      1. spaced`20150101000000 15343.231  3726.594 53053.606 .`
      2. glued  `20150101000000 -1143.350-46135.872-30139.569 .`   (WGRY 2015-05)
      3. glued  `20150101000000 14426.616   698.975200723.278 .`   (LGRR 2015-01..05)
      4. filled `20150101000000 99999.999838.800 99999.999 x`     (OXFO 2015-01..07)

    Forms 2 and 3 are whitespace failures: form 2 splits into 2 tokens instead of 5
    so X receives the timestamp, and form 3 glues Y onto Z when a value exactly
    fills its %9.3f field so Y is read as a concatenated value.  Form 4 is the
    station's own missing-value marker (99999.999 with a trailing 'x'), which must
    be caught PER ROW -- an OXFO day can start filled and recover mid-day, so a
    header-only check would let a mostly-filled day through as a fully
    interpolated series.  Every row therefore passes a |Z| > 40000 nT physical gate
    and a fill test, and an unusable day is DROPPED rather than filled with garbage.

    H is formed as sqrt(X^2 + Y^2) from the reported components rather than being
    read from the file, because CARISMA publishes cartesian XYZ only.  This is
    exact: the horizontal-plane norm is invariant under rotation, so it equals the
    true horizontal intensity whether the published axes are true-N/E or BGGF-N/E.

    The daily mean is then removed, which is what makes the network comparable with
    the ZW chain.  The Meridian "H" column is a DISTURBANCE series (MHT 2015 sits
    around -100 nT), whereas CARISMA reports the full field (ANNA ~+19000nT,
    RANK ~+3600 nT).  Eleven of the twelve entropy measures are exactly invariant
    under that offset -- verified numerically to 1e-9 for SE_full/SE_Pc3/SE_Pc4/
    SE_Pc5/PE/WPE/SampEn/Tsallis_q1.5/CH_C/CH_H/IR_ordinal -- but SVD entropy is
    NOT: a large DC offset becomes the leading singular value of the trajectory
    matrix and drives H_SVD to ~0 (measured 0.731 -> 0.016 with a 27k nT offset).
    Demeaning here makes the two networks differ only in where the signal comes
    from, not in the zero level, so no measure is silently degraded.

    Day acceptance matches the ZW/SuperMAG loaders: a day must have at least 50 %
    of its samples present, otherwise it is dropped rather than interpolated over a
    multi-hour hole (which would fabricate a smooth series and depress the band
    energies of that day).
    """
    sta_dir = Path(root) / station.lower() / str(year)
    files = sorted(sta_dir.glob("*/*.F01.gz"))
    if not files:
        raise SystemExit(f"no CARISMA 1 Hz files under {sta_dir}")
    out = []
    for f in files:
        # The date is only recoverable from the file NAME (YYYYMMDD...F01.gz); the
        # header carries it too but the name is the cheaper and safer source.
        stamp = f.name[:8]
        try:
            date = pd.to_datetime(stamp, format="%Y%m%d")
        except ValueError:
            continue
        df = _read_carisma_day(f)
        if df is None or df.empty:
            continue
        h = np.hypot(df["X"].to_numpy(dtype=float), df["Y"].to_numpy(dtype=float))
        if not np.isfinite(h).any():
            continue
        h = h - np.nanmean(h)      # disturbance level, matching the ZW dH column
        # Build the index from the file name + the file's own HH:MM:SS stamps
        # rather than from the first sample: a truncated day then lands on the
        # right wall-clock hours instead of being silently shifted to midnight.
        tod = pd.to_datetime(df["t"].str.slice(8, 14), format="%H%M%S",
                             errors="coerce")
        keep = tod.notna().to_numpy()
        if keep.sum() < 3600:
            continue
        idx = pd.to_datetime(date.strftime("%Y%m%d") + " " +
                             df.loc[keep, "t"].str.slice(8, 14),
                             format="%Y%m%d %H%M%S", errors="coerce")
        s = pd.Series(h[keep], index=idx)
        s = s[s.index.notna()]
        s = s[~s.index.duplicated()].sort_index()
        if s.empty or s.notna().sum() < 0.5 * 86400:
            continue
        s = s.reindex(pd.date_range(date, periods=86400, freq="s"))
        # Same interpolation as load_zw_raw(): the ordinal-pattern entropies are
        # corrupted by NaN inside a window even though the band energies are not.
        s = s.interpolate(limit_direction="both")
        out.append((date.date(), s))
    out.sort(key=lambda x: x[0])
    return out


# ------------------------------------------------------------ SuperMAG (1 min)
def load_sm(station, year, csv_path=None):
    f = Path(csv_path) if csv_path else SM_DIR / f"{station}_{year}_1min.csv.gz"
    df = pd.read_csv(f)
    col = "N_nez"
    if col not in df.columns:
        raise SystemExit(f"{f}: no N_nez column (got {list(df.columns)})")
    t = pd.to_datetime(df["time_utc"], utc=True)
    s = pd.Series(df[col].to_numpy(dtype=float), index=t)
    s = s[~s.index.duplicated()].sort_index()
    out = []
    for day, day_ser in s.groupby(s.index.date):
        day_ser = day_ser.asfreq("60s")
        if day_ser.notna().sum() < 0.5 * 1440:      # <12 h -> skip day
            continue
        day_ser = day_ser.interpolate(limit_direction="both")
        out.append((day, day_ser))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["zw", "zwraw", "zw2", "sm", "carisma"],
                    default="zw")
    ap.add_argument("--station", default="MHT")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--csv", default=None, help="explicit SM csv path (else auto)")
    ap.add_argument("--win", type=int, default=0,
                    help="window length in SAMPLES; 0 -> auto "
                         "(ZW 600 pts = 600 s @1Hz; SM 60 pts = 3600 s @1min)")
    ap.add_argument("--m-pe", type=int, default=0, help="PE/WPE order; 0 -> auto (zw 4 / sm 3)")
    ap.add_argument("--decimate", type=int, default=1,
                    help="ZW only: resample the 1-Hz series to N-second means "
                         "(e.g. 60 -> emulate 1-min data for an apples-to-apples "
                         "comparison with SuperMAG). Sets fs=1/N and the auto window.")
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--kp", type=float, default=0.0)
    ap.add_argument("--outdir",
                    default="/data/ssd-stor/fsf/workspaces/research_ulf/"
                            "ground_pc5/v8_entropy_dissipation/results")
    args = ap.parse_args()

    # CARISMA is natively 1 Hz, so it shares the ZW sampling rate and therefore
    # the ZW auto window (600 s) and PE order (m=4) -- that is the whole reason it
    # replaces SuperMAG, whose 1-min product forced a 1-h window and m=3.
    fs = 1.0 if args.src in ("zw", "zwraw", "zw2", "carisma") else 1.0 / 60.0
    if args.src in ("zw", "zwraw") and args.decimate > 1:
        fs = 1.0 / args.decimate          # emulate N-second data (match SuperMAG口径)
    bands = usable_bands(fs)
    # default window = 10 min at 1 Hz, else 1 h (units = SAMPLES)
    win = args.win or (600 if fs >= 1.0 else int(round(3600 * fs)))
    m_pe = args.m_pe or (4 if fs >= 1.0 else 3)
    edge = int(EDGE_SEC * fs)

    if args.src == "zw":
        days = load_zw(args.station, args.year)
    elif args.src == "zwraw":
        days = load_zw_raw(args.station, args.year)
    elif args.src == "zw2":
        days = load_zw2(args.station, args.year)
    elif args.src == "carisma":
        days = load_carisma(args.station, args.year)
    else:
        days = load_sm(args.station, args.year, args.csv)
    if args.src in ("zw", "zwraw") and args.decimate > 1:
        # boxcar to N-second means (closer to a real N-second instrument product
        # than plain subsampling, and avoids aliasing high-frequency content)
        step = "%ds" % args.decimate
        days = [(d, s.resample(step).mean()) for d, s in days]
    if args.max_days:
        days = days[:args.max_days]
    print(f"[mb] src={args.src} station={args.station} {args.year}: "
          f"{len(days)} days, win={win}s ({win * fs:.0f} pts), m_pe={m_pe}, fs={fs:g} Hz",
          flush=True)

    mlat = MLAT.get(args.station, 30.0)
    sig = dr.pedersen_conductance(mlat, args.kp)
    rows = []
    for dnum, (day, s) in enumerate(days, 1):
        raw = s.to_numpy()
        filt = {b: band_filter(raw, b, fs) for b in bands}
        for w0 in range(0, raw.size - win + 1, win):
            rw = raw[w0:w0 + win]
            if np.all(~np.isfinite(rw)):
                continue
            S_all, E_all, dB = dr.window_deposition(rw, mlat, kp=args.kp, fs=fs)
            ent = eb.entropy_battery(rw, fs=fs, m_pe=m_pe)
            row = {"time": s.index[w0], "day": str(day),
                   "Q_all": S_all, "E_int_all": E_all, "dB_rms": dB}
            inband = (w0 >= edge) and (w0 + win + edge <= raw.size)
            for b in bands:
                fw = filt[b][w0:w0 + win]
                if inband:
                    row[f"E_{b}"] = float(np.mean(fw ** 2))
                    row[f"P_{b}"] = float(np.mean(dr.poynting_flux(fw * 1e-9, sig)))
                else:
                    row[f"E_{b}"] = np.nan       # bandpass transient zone
                    row[f"P_{b}"] = np.nan
            for k in SCALAR_KEYS:
                row[k] = ent.get(k, np.nan)
            rows.append(row)
        if dnum % 30 == 0:
            print(f"[mb] ... day {dnum}/{len(days)}", flush=True)

    out = pd.DataFrame(rows)
    if out.empty or "E_Pc5" not in out.columns:
        print("[mb] no windows produced (rows=%d) -- check --win units (samples!)"
              % len(rows), flush=True)
        return
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"mb_{args.src}_{args.station}{args.year}_w{win}"
    csv_path = outdir / f"{tag}_multiband.csv"
    out.to_csv(csv_path, index=False)
    print(f"[mb] wrote {csv_path} ({len(out)} windows)", flush=True)

    # ---- stats per band ----
    ok = out["E_Pc5"].notna() & out["WPE"].notna()
    if ok.sum() > 30:
        sub = out[ok]
        print("[mb] band medians: " + "  ".join(
            f"{b}: E={sub['E_' + b].median():.3g} P={sub['P_' + b].median():.3g}"
            for b in bands), flush=True)
        for key in ["SE_full", "WPE", "CH_C", "IR_ordinal"]:
            line = f"[mb] {key:11s}"
            for b in bands:
                le = np.log10(sub[f"E_{b}"].clip(1e-12))
                r = sub[key].corr(le, method="spearman")
                line += f"  {b}={r:+.3f}"
            lq = np.log10(sub["Q_all"].clip(1e-30))
            line += f"  | Q_all={sub[key].corr(lq, method='spearman'):+.3f}"
            print(line, flush=True)

    # ---- figure: SE_full vs each band's energy ----
    if ok.sum() > 10:
        sub = out[ok]
        plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                             "xtick.labelsize": 10, "ytick.labelsize": 10,
                             "axes.titlesize": 11})
        fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 3.8))
        for ax, b in zip(np.atleast_1d(axes), bands):
            m = sub[f"E_{b}"].notna()
            ax.scatter(sub.loc[m, f"E_{b}"], sub.loc[m, "SE_full"],
                       s=9, alpha=0.35, c="steelblue", edgecolors="none")
            ax.set_xscale("log")
            ax.set_xlabel(rf"$E_{{{b}}}$ [nT$^2$]")
            r = sub.loc[m, "SE_full"].corr(np.log10(sub.loc[m, f"E_{b}"].clip(1e-12)),
                                           method="spearman")
            ax.set_title(f"SE_full vs {b}   ρ={r:+.3f}", fontsize=10)
            ax.grid(alpha=0.25, lw=0.4)
        axes[0].set_ylabel("SE_full")
        fig.suptitle(f"{args.station} {args.year}: spectral entropy vs band energy "
                     f"({args.src}, entropy window {win}s)", fontsize=12)
        fig.subplots_adjust(wspace=0.28, top=0.82)
        fig.savefig(outdir / f"{tag}_bands.png", dpi=130)
        plt.close(fig)
        print(f"[mb] figure -> {tag}_bands.png", flush=True)


if __name__ == "__main__":
    main()
