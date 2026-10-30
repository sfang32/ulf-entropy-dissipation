#!/bin/bash
# v3: regenerate Fig.7 from precomputed CSV (plot-only) + the 7 remaining figures.
set -u
R=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
cd "$R" || exit 1
PY=/data/ssd-stor/fsf/anaconda3/bin/python
mkdir -p logs

run () {
    local tag="$1"; shift
    echo "[r3] >>> $tag"
    if "$@" > "logs/r3_${tag}.log" 2>&1; then
        echo "[r3] OK   $tag"
    else
        echo "[r3] FAIL $tag (see logs/r3_${tag}.log)"
    fi
}

run tsallis  $PY code/tsallis_plotonly.py
run e4       $PY code/e4_storm.py
run mht      $PY code/plot_multiband.py --csv results/mb_zw_MHT2015_w600_multiband.csv --out results/mb_zw_MHT2015_bands_v2.png --metric SE_full
run e7       $PY code/e7_anchor.py
run repl     $PY code/fig_replication.py --out results/fig_replication.png
run e12      $PY code/e12_swarm_ground.py --crossings results/swarm_crossings_20150101_20151231.csv
run case_wpe $PY code/case_study_2015.py --start 2015-03-15 --end 2015-03-21 --metric WPE
run gate3    $PY code/fig_gate3_years.py

echo "[r3] DONE"
