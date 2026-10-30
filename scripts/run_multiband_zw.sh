#!/bin/bash
# v8 multi-band (direction C) full batch: 9 ZW stations x 2015, 4-way parallel.
# Runs alongside the Pc5-only batch (different output tag mb_*, no clash).
PY=/data/ssd-stor/fsf/anaconda3/bin/python
CODE=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation/code
cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
mkdir -p logs
echo "===== [$(date '+%F %T')] MULTIBAND BATCH START (9 stations) ====="
printf '%s\n' MHT MZL NAT SSL MLS JFT LAT PXT SYS \
  | xargs -P 4 -I{} sh -c "$PY $CODE/run_multiband.py --src zw --station {} --year 2015 > logs/mb_zw_{}.log 2>&1 && echo '[done] {}' || echo '[FAIL] {}'"
echo "===== [$(date '+%F %T')] MULTIBAND BATCH DONE ====="
