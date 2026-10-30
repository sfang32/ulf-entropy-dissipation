"""audit_e2.py -- verify every headline number quoted in manuscript_v8.md §3.2 / §3.10.

Runs on the server against the 16 e2_sigma per-dataset CSVs
(e2_sigma_zw2011..zw2022, zw22025, zw22026, zwraw2023, zwraw2024, e2_sigma.csv = Phase-I 2015).

Checks
------
1. D1 station-mean partial rho (band Pc5) under the AE scheme   -> §3.2 table
2. same under the PC scheme                                      -> §3.2 table
3. const scheme is identically zero (null model)                 -> §3.2 table third column
4. per-year station means 2011-2024                              -> §3.2 year table
5. across 16 data sets: mean, sd, min, max, counts above 0.2 / 0.1 / 0
                                                                 -> §3.2 prose + §3.10 table
"""
from __future__ import annotations

import glob
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")

FILES = ["e2_sigma.csv"] + sorted(
    os.path.basename(f) for f in glob.glob(os.path.join(RESULTS, "e2_sigma_*.csv"))
)


def load(path):
    out = []
    with open(path) as fh:
        head = fh.readline().strip().split(",")
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 7 or not p[1] or not p[2]:
                continue
            rec = dict(zip(head, p))
            try:
                rec["p"] = float(rec.get("p_P_given_E") or 0.0)
            except ValueError:
                rec["p"] = 0.0
            out.append(rec)
    return out


# ---------------------------------------------------------------- 1-3: D1 table
print("=" * 78)
print("1-3. D1 (Phase-I 2015) station-mean partial rho, band Pc5")
print("=" * 78)
d1 = [r for r in load(os.path.join(RESULTS, "e2_sigma.csv")) if r["band"] == "Pc5"]
mean = defaultdict(lambda: defaultdict(list))
for r in d1:
    if r["scheme"] in ("AE", "PC"):
        mean[r["scheme"]]["const" if r["scheme"] == "AE" else "AE"][r["metric"]].append(r["p"]) \
            if False else None
# build cleanly
byscheme = defaultdict(lambda: defaultdict(list))
for r in d1:
    byscheme[r["scheme"]][r["metric"]].append(r["p"])

QUOTE = {
    "SampEn": (0.236, 0.149),
    "SE_full": (0.205, 0.132),
    "WPE": (0.073, 0.013),
    "Tsallis_q1.5": (-0.101, -0.053),
    "CH_C": (-0.114, -0.056),
    "IR_ordinal": (-0.123, -0.073),
}
print("%-16s %11s %11s %11s | %11s %11s" % ("metric", "AE(csv)", "AE(paper)", "diff",
                                            "PC(csv)", "PC(paper)"))
for m, (ae, pc) in QUOTE.items():
    a = st.fmean(byscheme["AE"][m]) if byscheme["AE"][m] else float("nan")
    c = st.fmean(byscheme["PC"][m]) if byscheme["PC"][m] else float("nan")
    print("%-16s %11.4f %11.3f %11.4f | %11.4f %11.3f" % (m, a, ae, a - ae, c, pc))
print("\nconst (null model) station-mean, per metric  [must be ~0]:")
for m in ("SampEn", "SE_full", "WPE", "Tsallis_q1.5", "CH_C", "IR_ordinal"):
    v = byscheme.get("const", {}).get(m, [])
    print("   %-16s mean=%+.6f  max|.|=% .2e" % (m, st.fmean(v), max(abs(x) for x in v)))

# --------------------------------------------------------------- 4: per-year
print()
print("=" * 78)
print("4. Per-year station mean, band Pc5, AE scheme  (paper table)")
print("=" * 78)
YEARS = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018",
         "2019", "2020", "2021", "2022", "2023", "2024"]
PAPER_YEAR = {
    "2011": (0.154, 0.174, 0.072), "2012": (0.188, 0.201, 0.042),
    "2013": (0.183, 0.196, 0.080), "2014": (0.143, 0.142, 0.066),
    "2015": (0.205, 0.236, 0.073), "2016": (0.222, 0.229, 0.080),
    "2017": (0.240, 0.228, 0.040), "2018": (0.224, 0.204, 0.015),
    "2019": (0.255, 0.240, -0.008), "2020": (0.192, 0.179, 0.005),
    "2021": (0.193, 0.201, -0.008), "2022": (0.215, 0.214, -0.035),
    "2023": (0.183, 0.229, -0.011), "2024": (0.144, 0.176, -0.011),
}
allsets = defaultdict(list)
hdr = "%-6s %9s %9s %9s | %9s %9s %9s" % ("year", "SE_full", "SampEn", "WPE",
                                           "PH", "PS", "PW")
print(hdr)
bad = 0
for y in YEARS:
    cand = [os.path.join(RESULTS, f) for f in FILES if f.endswith("zw%s.csv" % y)
            or f.endswith("zwraw%s.csv" % y)]
    if not cand:
        print("%-6s  -- file missing --" % y)
        continue
    recs = [r for r in load(cand[0]) if r["band"] == "Pc5" and r["scheme"] == "AE"]
    vals = {m: [r["p"] for r in recs if r["metric"] == m] for m in ("SE_full", "SampEn", "WPE")}
    got = tuple(st.fmean(vals[m]) for m in ("SE_full", "SampEn", "WPE"))
    want = PAPER_YEAR[y]
    for g, w in zip(got, want):
        if abs(round(g, 3) - w) > 1e-9:
            bad += 1
    print("%-6s %9.4f %9.4f %9.4f | %9.3f %9.3f %9.3f"
          % (y, got[0], got[1], got[2], want[0], want[1], want[2]))
    for m in ("SE_full", "SampEn", "WPE"):
        allsets[m] += vals[m]
print("per-year mismatches vs paper:", bad)

# ------------------------------------------------------- 5: across 16 datasets
print()
print("=" * 78)
print("5. Across the 16 data sets (14 Phase-I years + 2025 + 2026)")
print("=" * 78)
PA = {"SE_full": 0.194, "SampEn": 0.206, "WPE": 0.029}
counts = {"SE_full": [0.2, 0.1, 0.0], "SampEn": [0.2, 0.1, 0.0], "WPE": [0.1]}
for m in ("SampEn", "SE_full", "WPE", "CH_C", "Tsallis_q1.5"):
    v = allsets.get(m, [])
    if not v:
        print("  %-14s (per-year files only)" % m)
        continue
    print("%-14s mean=%+.4f sd=%.4f  min=%+.4f  max=%+.4f  n=%d"
          % (m, st.fmean(v), st.stdev(v), min(v), max(v), len(v)))
print()
print("counts (paper: SampEn clears 0.2 in 10/16 and +0.10 in 16/16;")
print("        SE_full clears 0.2 in 6/16 and +0.10 in 16/16;")
print("        WPE clears +0.10 in none of the sixteen)")
for m, thr in (("SampEn", 0.2), ("SE_full", 0.2)):
    v = allsets[m]
    print("  %-9s > %.1f : %d/16   > 0.1 : %d/16   > 0 : %d/16"
          % (m, thr, sum(1 for x in v if x > thr),
             sum(1 for x in v if x > 0.1), sum(1 for x in v if x > 0)))
v = allsets.get("WPE", [])
print("  %-9s > 0.1 : %d/16" % ("WPE", sum(1 for x in v if x > 0.1)))
