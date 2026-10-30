# ulf-entropy-dissipation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23797833.svg)](https://doi.org/10.5281/zenodo.23797833)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Analysis code for **Ground-magnetometer entropy as a broadband indicator of magnetosphere–ionosphere
energy dissipation** (Fang & Zou, submitted to *Entropy*, MDPI).

**Companion dataset:** [10.57760/sciencedb.space.0034c](https://doi.org/10.57760/sciencedb.space.0034c)
— 321 files, 4.3 GB of per-window ULF statistics, sensitivity runs, summary tables and station metadata,
2011–2026, 65 stations.

**This repository contains no data.** It holds the 63 scripts (47 Python, 15 shell, 1 draw.io source)
that turn 1 Hz ground magnetometer records into every number and every figure of the paper.

## Contents

| Directory | Files | Contents |
|---|---|---|
| `core/` | 4 | The library everything imports: an eleven-measure entropy/complexity battery (numpy + scipy only), Pedersen-conductance and Poynting-flux schemes, true AACGM-v2 conversion, and the per-window driver |
| `experiments/` | 28 | One script per experiment (E1…E14) plus the number auditors |
| `figures/` | 6 | Fig. 1 source (`fig1-workflow.drawio` + `drawio2svg.py`) and the four `*_plotonly.py` re-plotters |
| `acquisition_qc/` | 10 | Probes and QC for CARISMA, the Meridian raw DMD product, OMNI and SuperMAG |
| `scripts/` | 15 | The production drivers, verbatim as run |
| `docs/SCRIPTS.md` | 1 | Script-by-script reference: what each file reads, writes, and which figure or table it produces |

## Install

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/core:$PYTHONPATH"    # the experiments import core/ modules directly
```

Reference environment that produced the published results: Python 3.10.13, numpy 1.26.4, scipy 1.11.4,
pandas 2.1.4, matplotlib 3.8.3, aacgmv2 2.7.1, cdflib 1.3.1, Pillow 12.2.

No third-party entropy library is used — `ordpy`, `antropy` and `infomeasure` are absent on purpose, so
no number depends on another package's estimator conventions.

## Quick start

The two `*_plotonly.py` scripts read only the deposited summary tables, so they run in seconds with no
raw data and no OMNI:

```bash
DATA=/path/to/entropy_ulf_dissipation_v8      # unpacked data deposit
cp "$DATA/01_DATA/03_summary_tables/e2_sigma_zw2015.csv" \
   "$DATA/01_DATA/03_summary_tables/tsallis_scan.csv" /tmp/
# point the RES constant at those two files, then
MPLBACKEND=Agg python figures/e2_sigma_plotonly.py
MPLBACKEND=Agg python figures/tsallis_plotonly.py
```

This has been run against the published deposit: both scripts exit 0 and redraw Fig. 4 and Fig. 14 from
the deposited CSVs alone.

## Running the full chain

The scripts were written for one machine and carry **seven absolute roots** (documented in
`docs/SCRIPTS.md` § *Re-pointing the hard-coded roots*). They were left as-run rather than parametrised,
so that the deposited files are the ones that actually produced the numbers. A blunt port:

```bash
OLD=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
NEW=/path/to/your/results/tree
grep -rl "$OLD" --include=*.py . | xargs sed -i "s|$OLD|$NEW|g"
```

Order: `core/run_multiband.py` (per-window products) → `experiments/*` (statistics) →
`figures/*` (artwork). `scripts/rebuild_figs*.sh` show the dependency order used for the final figures.

## Figure → script

| Fig. | Script | Reads |
|---|---|---|
| 1 | `figures/fig1-workflow.drawio` → `drawio2svg.py` | — (schematic) |
| 2 | `figures/fig_replication.py` | per-window CSVs |
| 3 | `figures/fig_gate3_years.py` | per-window CSVs, 16 data sets |
| 4 | `experiments/e2_sigma.py` / `figures/e2_sigma_plotonly.py` | `e2_sigma.csv` |
| 5, 10 | `experiments/mb_summary.py` | per-window CSVs |
| 6 | `experiments/plot_multiband.py` | one station-year |
| 7 | `experiments/e12_swarm_ground.py` (+ `swarm_joule.py`) | Swarm L1b, per-window CSVs |
| 8 | `experiments/case_study_2015.py` | Swarm L1b, per-window CSVs, OMNI |
| 9 | `experiments/sens_win9.py` | `02_sensitivity_runs/` |
| 11 | `experiments/e5_latitude.py` | `mb_summary.csv` |
| 12 | `experiments/e4_storm.py` | per-window 2015 CSVs, OMNI |
| 13 | `experiments/e7_anchor.py` | per-window 2015 CSVs |
| 14 | `experiments/tsallis_scan.py` / `figures/tsallis_plotonly.py` | `tsallis_scan.csv` |

## Notes and caveats

1. **`*_plotonly.py` are not byte-identical to the published PNGs.** Fig. 4 and Fig. 14 were re-styled
   for MDPI after the statistics were frozen; these variants read the archived CSV and redraw the same
   numbers in a simpler style.
2. **`acquisition_qc/pull_e5b_2015.py` needs your own SuperMAG account** — export `SUPERMAG_LOGON=<name>`.
   `supermag_api` itself is not bundled.
3. **`figures/drawio2svg.py` carries a Windows-first font list** with fallbacks; opening the `.drawio`
   in the draw.io desktop app avoids it.
4. **`SampEn` returns NaN, not 0,** when the *m*+1 template count is zero. Filling those gaps silently
   biases the low-activity tail.
5. **The OMNI fill value is ≈999999.9**, which a `|x| > 1e30` mask does *not* catch. `omni_2015_ascii.py`
   masks it properly; keep that masking if you re-read OMNI.

## Licence

MIT — see `LICENSE`. The data are released under CC BY 4.0.

## Citation

> Fang, S.; Zou, Z. *Ground-magnetometer entropy as a broadband indicator of magnetosphere–ionosphere
> energy dissipation*. Submitted to Entropy (MDPI), 2026.

For the numbers, cite the dataset ([`10.57760/sciencedb.space.0034c`](https://doi.org/10.57760/sciencedb.space.0034c));
for the pipeline, cite the archived software:

> Fang, S.; Zou, Z. *Analysis code for ground-magnetometer ULF entropy and dissipation indicators*,
> v1.0.0. Zenodo, 2026. <https://doi.org/10.5281/zenodo.23797833>

The concept DOI [`10.5281/zenodo.23797832`](https://doi.org/10.5281/zenodo.23797832) always resolves to
the latest version; the version DOI above pins `v1.0.0`, which is what the paper cites.
