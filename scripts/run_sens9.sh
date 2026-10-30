#!/bin/bash
# run_sens9.sh -- matched-station sampling comparison.
#
# The original §3.7 sampling table compared 1 Hz / 60 min over {MHT,NAT,SYS}
# with 1 min / 60 min over {MHT,NAT,MLS}: two station sets, so the sampling
# effect was confounded with a station change.  This batch produces both rows on
# one identical nine-station set; sens_pair.py then reads them.
PY=/data/ssd-stor/fsf/anaconda3/bin/python
cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
mkdir -p logs results/sens9
echo "===== [$(date '+%F %T')] sens9 batch (9 stations x 2 cadences) ====="
for s in MHT MZL NAT SSL MLS JFT LAT PXT SYS; do
  echo "$s hz"
  echo "$s min"
done | xargs -P 4 -n 2 sh -c '
  ST="$0"; MODE="$1"
  PY=/data/ssd-stor/fsf/anaconda3/bin/python
  cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
  if [ "$MODE" = "hz" ]; then
    $PY code/run_multiband.py --src zw --station "$ST" --year 2015 --win 3600 \
        --outdir results/sens9 > logs/sens9_${ST}_hz.log 2>&1 \
      && echo "[done] $ST 1Hz/60min" || echo "[FAIL] $ST 1Hz/60min"
  else
    $PY code/run_multiband.py --src zw --station "$ST" --year 2015 --decimate 60 \
        --outdir results/sens9 > logs/sens9_${ST}_min.log 2>&1 \
      && echo "[done] $ST 1min/60min" || echo "[FAIL] $ST 1min/60min"
  fi
'
echo "===== [$(date '+%F %T')] sens9 DONE ====="
