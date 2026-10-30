#!/bin/bash
# Resume figure rebuild from e2 onward.
set -u
R=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
cd "$R" || exit 1
PY=/data/ssd-stor/fsf/anaconda3/bin/python
mkdir -p logs

run () {
    local tag="$1"; shift
    echo "[rebuild] >>> $tag"
    if "$@" > "logs/rebuild_v2_${tag}.log" 2>&1; then
        echo "[rebuild] OK   $tag"
    else
        echo "[rebuild] FAIL $tag (see logs/rebuild_v2_${tag}.log)"
    fi
}

# e2 was stuck on OMNI CDF read; use existing CSV and redraw only.
run e2po $PY code/e2_sigma_plotonly.py
run tsallis $PY code/tsallis_scan.py --stations MHT MZL NAT SSL MLS JFT LAT PXT SYS
run e4 $PY code/e4_storm.py
run mht2015 $PY code/plot_multiband.py --csv results/mb_zw_MHT2015_w600_multiband.csv --out results/mb_zw_MHT2015_bands_v2.png --metric SE_full
run e7 $PY code/e7_anchor.py
run repl $PY code/fig_replication.py --out results/fig_replication.png
run e12 $PY code/e12_swarm_ground.py --crossings results/swarm_crossings_20150101_20151231.csv
run case_wpe $PY code/case_study_2015.py --start 2015-03-15 --end 2015-03-21 --metric WPE
run gate3 $PY code/fig_gate3_years.py

echo "[rebuild] RESUME DONE"
