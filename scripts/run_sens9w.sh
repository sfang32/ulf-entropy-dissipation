#!/bin/bash
# run_sens9w.sh -- window-length sensitivity on the full nine-station set.
#
# The original sensitivity batch (run_sens.sh) used three stations chosen for
# latitude spread.  §3.6 quotes the resulting Pc5 sign reversal as a headline
# robustness result, so this batch repeats the 20- and 40-minute windows on all
# nine Phase-I stations; the 60-minute row already exists in results/sens9.
PY=/data/ssd-stor/fsf/anaconda3/bin/python
cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
mkdir -p logs results/sens9w
echo "===== [$(date '+%F %T')] sens9w batch (9 stations x {1200,2400}) ====="
for s in MHT MZL NAT SSL MLS JFT LAT PXT SYS; do
  echo "$s 1200"
  echo "$s 2400"
done | xargs -P 4 -n 2 sh -c '
  ST="$0"; WIN="$1"
  PY=/data/ssd-stor/fsf/anaconda3/bin/python
  cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
  $PY code/run_multiband.py --src zw --station "$ST" --year 2015 --win "$WIN" \
      --outdir results/sens9w > logs/sens9w_${ST}_w${WIN}.log 2>&1 \
    && echo "[done] $ST w$WIN" || echo "[FAIL] $ST w$WIN"
'
echo "===== [$(date '+%F %T')] sens9w DONE ====="
