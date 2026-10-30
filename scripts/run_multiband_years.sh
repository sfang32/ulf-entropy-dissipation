#!/bin/bash
# ZW (Meridian-I FGM) 9 stations x 2011-2022 (11 extra years; 2015 already done).
# 6-way parallel, ~10 h total (overnight). Output: mb_zw_{ST}{YYYY}_multiband.csv
PY=/data/ssd-stor/fsf/anaconda3/bin/python
CODE=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation/code
cd /data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation || exit 1
mkdir -p logs
echo "===== [$(date '+%F %T')] MULTI-YEAR BATCH START (9 stations x 11 years) ====="
for s in MHT MZL NAT SSL MLS JFT LAT PXT SYS; do
  for y in 2011 2012 2013 2014 2016 2017 2018 2019 2020 2021 2022; do
    echo "$s $y"
  done
done | xargs -P 6 -n 2 sh -c "$PY $CODE/run_multiband.py --src zw --station \$0 --year \$1 > logs/mby_\$0_\$1.log 2>&1 && echo \"[done] \$0 \$1\" || echo \"[FAIL] \$0 \$1\""
echo "===== [$(date '+%F %T')] MULTI-YEAR BATCH DONE ====="
