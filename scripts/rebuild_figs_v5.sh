#!/bin/bash
# v5: second-pass polish after Qwen vision re-review of the v4 renders.
#   fig3  ylim headroom + legend inside upper-right
#   fig4  right-panel legend inside upper-left + tighter left xlim
#   fig5  add "individual stations" legend entry to panel 4
#   fig7  drop supxlabel, legend to top, Y ticks on every panel
#   fig9  add (a)/(b) panel titles
#   fig12 legend to bottom
#   fig13 remove diagonal highlight boxes, colorbar ticks, contrast 0.8->0.6
#   fig14 restore rho/N in panel titles, Y grid, legend to top
# Same chain as v4 (mb_summary --include-sm first so fig9 keeps the 3 high-lat SuperMAG sites).
set -u
R=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
cd "$R" || exit 1
PY=/data/ssd-stor/fsf/anaconda3/bin/python
mkdir -p logs

run () {
    local tag="$1"; shift
    echo "[r5] >>> $tag"
    if "$@" > "logs/r5_${tag}.log" 2>&1; then
        echo "[r5] OK   $tag"
    else
        echo "[r5] FAIL $tag (see logs/r5_${tag}.log)"
    fi
}

run mb_summary $PY code/mb_summary.py --year 2015 --include-sm

run e5       $PY code/e5_latitude.py
run e12      $PY code/e12_swarm_ground.py --crossings results/swarm_crossings_20150101_20151231.csv
run case_wpe $PY code/case_study_2015.py --start 2015-03-15 --end 2015-03-21 --metric WPE
run sens     $PY code/sens_win9.py
run tsallis  $PY code/tsallis_plotonly.py
run mht      $PY code/plot_multiband.py --csv results/mb_zw_MHT2015_w600_multiband.csv --out results/mb_zw_MHT2015_bands_v2.png --metric SE_full
run gate3    $PY code/fig_gate3_years.py

echo "[r5] DONE"
