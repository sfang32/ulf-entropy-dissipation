#!/bin/bash
# Window-length sensitivity (direction C robustness check).
# ZW 1-Hz, 3 stations (high/mid/low latitude) x 4 entropy windows:
#   600 s (10 min, baseline), 1200 (20), 2400 (40), 3600 (60)
# Output files carry _w{win} so nothing is overwritten.
PY=/data/ssd-stor/fsf/anaconda3/bin/python
CODE=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation/code
cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
mkdir -p logs
echo "===== [$(date '+%F %T')] WINDOW-SENSITIVITY BATCH (3 stations x 4 windows) ====="
for s in MHT NAT SYS; do
  for w in 600 1200 2400 3600; do
    echo "$s $w"
  done
done | xargs -P 4 -n 2 sh -c "$PY $CODE/run_multiband.py --src zw --station \$0 --year 2015 --win \$1 > logs/sens_\$0_w\$1.log 2>&1 && echo \"[done] \$0 w\$1\" || echo \"[FAIL] \$0 w\$1\""
echo "===== [$(date '+%F %T')] SENSITIVITY DONE ====="
