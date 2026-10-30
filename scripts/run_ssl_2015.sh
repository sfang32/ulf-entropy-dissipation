#!/bin/bash
# SSL (Beijing Shisanling) recovery run: in parallel with the main batch (64-core box)
PY=/data/ssd-stor/fsf/anaconda3/bin/python
CODE=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation/code
echo "===== [$(date '+%F %T')] station SSL year 2015 (v2 band-energy, SampEn-fix, mlat=30.8) ====="
$PY $CODE/run_prototype.py --station SSL --year 2015
echo "===== [$(date '+%F %T')] SSL_DONE ====="
