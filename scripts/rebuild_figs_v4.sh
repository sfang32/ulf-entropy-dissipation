#!/bin/bash
# v4: fix user-flagged visual issues (in-figure text, legend overlap, missing legends,
#     significance stars, truncated titles) and regenerate SuperMAG-included E5.
set -u
R=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
cd "$R" || exit 1
PY=/data/ssd-stor/fsf/anaconda3/bin/python
mkdir -p logs

run () {
    local tag="$1"; shift
    echo "[r4] >>> $tag"
    if "$@" > "logs/r4_${tag}.log" 2>&1; then
        echo "[r4] OK   $tag"
    else
        echo "[r4] FAIL $tag (see logs/r4_${tag}.log)"
    fi
}

# Rebuild summary with SuperMAG so E5 latitude panel actually shows the 3 high-lat stations.
run mb_summary $PY code/mb_summary.py --year 2015 --include-sm

run e5       $PY code/e5_latitude.py
run e12      $PY code/e12_swarm_ground.py --crossings results/swarm_crossings_20150101_20151231.csv
run case_wpe $PY code/case_study_2015.py --start 2015-03-15 --end 2015-03-21 --metric WPE
run sens     $PY code/sens_win9.py
run tsallis  $PY code/tsallis_plotonly.py
run mht      $PY code/plot_multiband.py --csv results/mb_zw_MHT2015_w600_multiband.csv --out results/mb_zw_MHT2015_bands_v2.png --metric SE_full
run gate3    $PY code/fig_gate3_years.py

echo "[r4] DONE"
