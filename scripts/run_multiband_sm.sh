#!/bin/bash
# E5b: SuperMAG 1-min (KIR/YKC/BRW) -> multi-band pipeline, 3600 s window, m_pe=3
PY=/data/ssd-stor/fsf/anaconda3/bin/python
CODE=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation/code
cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
mkdir -p logs
echo "===== [$(date '+%F %T')] E5b (SuperMAG 1-min) MULTIBAND BATCH ====="
printf '%s\n' KIR YKC BRW \
  | xargs -P 3 -I{} sh -c "$PY $CODE/run_multiband.py --src sm --station {} --year 2015 > logs/mb_sm_{}.log 2>&1 && echo '[done] {}' || echo '[FAIL] {}'"
echo "===== [$(date '+%F %T')] E5b BATCH DONE ====="
