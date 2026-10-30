# Script reference — what each file does, and how to run it

> These scripts read the **per-window products** and **summary tables** of the companion data
> deposit (`10.57760/sciencedb.space.0034c`). Where the text below writes `01_DATA/…`, that is the
> data deposit's tree: `01_DATA/01_per_window/`, `01_DATA/03_summary_tables/`,
> `01_DATA/04_station_metadata/`. No data file is redistributed inside this code package.

## Environment that produced the published results

```
Python 3.10.13
numpy 1.26.4 · scipy 1.11.4 · pandas 2.1.4 · matplotlib 3.8.3
cdflib 1.3.x · aacgmv2 2.7.1 · Pillow 12.x (Fig. 1 rendering only)
```

`python -m pip install numpy scipy pandas matplotlib aacgmv2 cdflib pillow` reproduces this set to
within the patch level; nothing in the pipeline is sensitive to the minor version.

No `ordpy`, `antropy`, `infomeasure` or any other entropy package is used — every measure is
implemented in `core/entropy_battery.py` with numpy/scipy only, so the numbers do not depend on
a third-party estimator's conventions. `Pillow` is needed only if you want to re-render Fig. 1.

> A separate Python 3.11 interpreter exists on the machine that ran the production jobs; its
> matplotlib install is broken (`GLIBCXX_3.4.29` missing) and it was **not** used for any figure.
> If you reproduce on that interpreter and matplotlib fails to import, that is why.

---

## core/ — the four files everything else imports

| File | Role |
|---|---|
| `entropy_battery.py` | The 11-measure battery. `battery(x, m_pe=4)` returns a dict of `PE, WPE, SampEn, SE_*, Tsallis_q1.5, CH_H, CH_C, SVD, IR_ordinal`. Sample entropy is cKDTree-vectorised; `SampEn` returns NaN (not 0) when the *m*+1 template count is 0 — this distinction matters, a 0 would be a fake "low entropy" window. |
| `deposition_rate.py` | Pedersen conductance schemes (`const` = 5 S, `AE` = 5·√(AE/100), `PC` = 5·√(PC/1)) and the Poynting-flux proxy `P_b = mean(dB_b²/(μ0² Σ_P))`. |
| `compute_aacgm_v8.py` | True AACGM-v2 conversion via `aacgmv2`, epoch fixed at 2015-07-01 00:00 UT (Phase-I/CARISMA) or 2025-07-01 00:00 UT (Phase-II). Writes the `*_aacgm.csv` tables deposited in `01_DATA/04_station_metadata/` of the data package. |
| `run_multiband.py` | The per-window driver. Reads a station-year of 1 Hz data, filters the **whole daily series** with a 10th-order Butterworth `sos` (never per window), cuts non-overlapping windows, drops 1800 s at each day edge, evaluates the battery on the raw window, and writes `mb_<src>_<ST><YR>_w<W>_multiband.csv`. |

`run_multiband.py` options that matter:

```bash
--src {zw,zw2,zwraw,carisma,sm}   # data family
--station MHT --year 2015        # one station-year per run
--win 600                        # window length [s]
--m-pe 4                         # permutation-entropy order (4 at 1 Hz; 3 for the 1-min arm)
--decimate 60                    # emulate 1-min sampling from 1 Hz  (the dec60/ runs)
```

---

## experiments/ — one script per experiment (E1…E14) plus the auditors

| Script | Experiment | Reads | Writes | Figure / Table |
|---|---|---|---|---|
| `run_prototype.py` | E1 prototype | raw ZW series | `v2_<ST><YR>_pc5energy.csv` | provenance of §3.1 |
| `e1_summary.py` | E1 | `v2_*_pc5energy.csv` | console table | §3.1 |
| `e2_sigma.py` | E2 / E6 conductance schemes, degeneracy | `mb_*.csv` | `e2_sigma.csv`, `e2_sigma.png` | Fig. 4 |
| `e2_sigma_multiyear.py` | E2 across 16 data sets | `mb_*.csv` | `e2_sigma_zw*.csv`, `e2_multiyear_*.csv` | §3.2 |
| `e3_partial.py` | E3 partial correlation, energy removed | `mb_*.csv` | `e3_partial.csv` | §3.2 |
| `e3b_omni.py` | E3b external index reproduces the coefficient | `mb_*.csv` + OMNI | `e3b_omni.csv` | §3.3 |
| `e4_storm.py` | E4 superposed epoch, 54 storms | `mb_zw_*2015*`, OMNI SYM-H/AE | `e4_storm_phase.csv`, `e4_storm.png` | Fig. 12 |
| `e5_latitude.py` | E5 latitude arm, 23 CARISMA stations at 1 Hz | `mb_summary.csv`, `mb_carisma_*` | `e5_latitude.png` | Fig. 11 |
| `e7_anchor.py` | E7 absolute-magnitude anchor | `mb_zw_*2015_w600*` | `e7_anchor.csv`, `e7_anchor.png` | Fig. 13 |
| `e12_swarm_ground.py` | E12 ground vs Swarm Joule heating, 2015 | Swarm L1b, `mb_zw_*2015*` | `e12_swarm_ground.csv`, `.png` | Fig. 7 |
| `e12_swarm_ground_2014.py` | E12 replication on 2014 | as above | `e12_swarm_ground_2014.csv` | §3.4 |
| `e12_dedup.py` | crossing de-duplication bookkeeping | `swarm_crossings_*.csv` | console counts | 22 523 → 22 465 |
| `sens_win9.py` | E10 window-length sensitivity | `sens9/`, `sens9w/`, `dec60/` | `sens_window9.csv`, `.png` | Fig. 9 |
| `sens_pair.py`, `sens_summary.py` | paired / summary views of the same sweep | as above | `sens_pair.csv`, `sens_window.csv` | §3.6 |
| `tsallis_scan.py` | E14 q and bin scan | `mb_zw_*2015_w600*` | `tsallis_scan.csv`, `.png` | Fig. 14 |
| `case_study_2015.py` | E8 March-2015 case study | Swarm + `mb_zw_*2015*` | `case_2015_joint_*.csv`, `case_2015_storm_<metric>.png` | Fig. 8 |
| `mb_summary.py` | ρ per station × band × metric | `mb_*.csv` | `mb_summary.csv`, `mb_matrix*.csv`, `mb_heatmap.png`, `mb_corr_vs_mlat.png` | Figs. 5, 10 |
| `plot_multiband.py` | per-station band scatter | `mb_zw_MHT2015_w600*` | `mb_zw_MHT2015_bands.png` | Fig. 6 |
| `station_consistency.py` | cross-station sign agreement | `mb_summary.csv` | `station_consistency.csv` | §3.10 |
| `swarm_joule.py`, `swarm_timeseries.py` | Swarm-derived Joule heating | Swarm L1b MAG/EFI | `swarm_joule_A_*.csv` | Fig. 7 input |
| `diag_e4.py` | diagnostics for the epoch stack | `e4_storm_phase.csv` | console | §3.8 |
| `audit_e2.py`, `audit_e3.py` | number reconciliation of Tables 3–4 | summary CSVs | console PASS/FAIL | editorial audit |
| `make_tables.py` | Tables 1–3 | summary CSVs + station tables | Markdown | Tab. 1–3 |
| `make_supplementary.py` | Tables S1–S2 | summary CSVs | Markdown | Supp. |
| `omni_2015_ascii.py` | OMNI reader with correct fill-value masking | ASCII OMNI | arrays | E4/E7/e3b |

`e5_latitude.py` writes **only** the PNG: Fig. 11 is plotted straight from `mb_summary.csv`, so the
figure's numbers are the `rho` column of that table for the CARISMA rows.

---

## figures/

| File | Produces |
|---|---|
| `fig1-workflow.drawio` + `drawio2svg.py` | Fig. 1 (draw.io source → PNG/PDF/SVG, no GUI needed) |
| `fig_replication.py` | Fig. 2 |
| `fig_gate3_years.py` | Fig. 3 |
| `e2_sigma_plotonly.py` | Fig. 4, re-plotted from the pinned `e2_sigma.csv` |
| `tsallis_plotonly.py` | Fig. 14, re-plotted from the pinned `tsallis_scan.csv` |

`*_plotonly.py` exist because Fig. 4 and Fig. 14 were re-styled for MDPI without re-running the
statistics: they read the archived CSV, so the plotted numbers cannot drift from the deposited ones.
They re-draw the **same numbers in a slightly simpler style** and are *not* byte-identical to the
published PNG — the published Fig. 4 came from a full run of `e2_sigma.py`. Use them to check that
the deposited tables support the published panels; use the full scripts to regenerate the artwork.

Smoke test run on 2026-10-30 inside the deposit tree: both `*_plotonly.py` execute against
`01_DATA/03_summary_tables/` alone (no upstream data, no OMNI) and emit a PNG in seconds.

The remaining ten figures are drawn by their experiment script (see the table above). There is no
separate "plot" layer — this was deliberate, so that a figure cannot be edited without its numbers.

---

## acquisition_qc/

`dl_omni_1min.py`, `pull_e5b_2015.py` (SuperMAG), `scan_zw2.py`, `scan_carisma_layout.py`,
`probe_zw_raw.py`, `probe_despike.py`, `probe_oxfo.py`, `verify_zwraw.py`, `tz_check.py`.

These are the probes that fixed three data-layer bugs worth knowing about if you rebuild the input:
CARISMA writes four file-format variants, including a fill value printed as `99999.999x`
(209 of 355 days at OXFO) and days whose components are glued together when a value fills its
field; the Meridian raw DMD daily file has no header line and its date must be taken from the
filename; and the OMNI fill value is ≈999999.9, which a `|x|>1e30` mask does **not** catch.

---

## scripts/ — the production drivers, verbatim

`run_multiband_years.sh` (12 Phase-I years), `run_multiband_zw2.sh` (Phase-II), `run_multiband_zw.sh`,
`run_multiband_sm.sh`, `run_batch_2015.sh`, `run_ssl_2015.sh`, `run_sens*.sh`, `rebuild_figs*.sh`
(re-run the whole figure set in dependency order). They are kept as-run, including the `screen -dmS`
invocations used for the multi-day batches.

---

## Re-pointing the hard-coded roots

The scripts were written for one machine and carry absolute paths. There are seven distinct roots;
replace them with your own tree and nothing else changes:

| Constant in the code | What it points at |
|---|---|
| `/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation/` | the results tree — set this to **this deposit** and every figure script runs |
| `/data/ssd-stor/fsf/databases/data_ulf/dataset_pc/ZW` | Phase-I processed 1 Hz waveforms (per month/day CSV) |
| `/data/ssd-stor/YLK/YLK-BUFFER/NSSC-FSF/B015/dataset/data/zw1_fgm_dat` | Phase-I raw daily `DMD L01 01D` files (2023–2024) |
| `/data/ZW/file/success` | Phase-II FGMM01 archive (one file = one UTC day, 86 400 rows at 1 Hz) |
| `/data/ftp/fsf/databases/data_yuliaoku/CARISMA/FGM/1Hz` | CARISMA 1 Hz XYZ |
| `/data/ftp/fsf/databases/data_omni/omni_1min` | OMNI HRO 1-minute CDFs |
| `/data/ftp/fsf/databases/opensource/SWARM` | Swarm L1b MAG/EFI |

A blunt but effective port:

```bash
OLD=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
NEW=/your/path/to/entropy_ulf_dissipation_v8
grep -rl "$OLD" --include=*.py . | xargs sed -i "s|$OLD|$NEW|g"
```

Two files carry machine-specific defaults that are **not** roots, and both are overridable:

| File | What to set |
|---|---|
| `acquisition_qc/pull_e5b_2015.py` | SuperMAG needs a registered user name as its logon argument; export `SUPERMAG_LOGON=<your name>` (free registration at supermag.jhuapl.edu) and optionally `SUPERMAG_E5B_OUT`. The account name is deliberately not hard-coded in this deposit. |
| `figures/drawio2svg.py` | Font list is Windows-first (`C:/Windows/Fonts/arial*.ttf`, `segoeui*`, `msyh`) with fallbacks; on Linux it falls through to whatever the fallback chain finds. Re-rendering Fig. 1 from `fig1-workflow.drawio` in the draw.io desktop app avoids the issue entirely. |

`superMag_api` itself is **not** bundled — download it from SuperMAG if you re-pull the E5b
1-minute comparison stations; nothing else in the pipeline needs it.
