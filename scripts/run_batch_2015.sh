#!/bin/bash
# v8 batch v2 rerun (SampEn A<=0 -> nan fix): remaining 6 stations only
# MHT 2015 already completed with old code (no A=0 hit -> results valid; new code identical for A>0)
PY=/data/ssd-stor/fsf/anaconda3/bin/python
CODE=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation/code
for s in MZL MLS JFT LAT PXT SYS; do
  echo "===== [$(date '+%F %T')] station $s year 2015 (v2 band-energy, SampEn-fix) ====="
  $PY $CODE/run_prototype.py --station $s --year 2015
done
echo "===== [$(date '+%F %T')] BATCH_DONE ====="
