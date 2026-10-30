#!/bin/bash
# Rebuild every figure that the Qwen vision review flagged, in one pass.
# Covers: Fig.3 (mb_corr_vs_mlat), Fig.6 (e2_sigma), Fig.7 (tsallis_scan),
#         Fig.8 (e4_storm), Fig.10 (e7_anchor), Fig.12 (e12), Fig.13 (case study).
set -u
R=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
cd "$R" || exit 1
PY=/data/ssd-stor/fsf/anaconda3/bin/python
mkdir -p logs

run () {   # run <log-tag> <cmd...>
    local tag="$1"; shift
    echo "[rebuild] >>> $tag"
    if "$@" > "logs/rebuild_${tag}.log" 2>&1; then
        echo "[rebuild] OK   $tag"
    else
        echo "[rebuild] FAIL $tag (see logs/rebuild_${tag}.log)"
    fi
}

# Fig.3 heatmap + latitude panels (clean 9-station phase-I data)
run mb2015 $PY code/mb_summary.py --year 2015
# Fig.6 conductance schemes (reads the full-year OMNI CDFs -- the slow one)
run e2 $PY code/e2_sigma.py
# Fig.10 absolute anchor
run e7 $PY code/e7_anchor.py
# Fig.7 Tsallis parameter scan
run tsallis $PY code/tsallis_scan.py
# Fig.8 storm superposed epoch
run e4 $PY code/e4_storm.py
# Fig.12 ground-space coupling (reuse the year-long crossings table)
run e12 $PY code/e12_swarm_ground.py \
    --crossings results/swarm_crossings_20150101_20151231.csv
# Fig.13 storm case study, both metrics
run case_wpe $PY code/case_study_2015.py --start 2015-03-15 --end 2015-03-21 --metric WPE
run case_se  $PY code/case_study_2015.py --start 2015-03-15 --end 2015-03-21 --metric SE_full

echo "[rebuild] ALL DONE"
