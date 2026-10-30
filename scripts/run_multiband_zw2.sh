#!/bin/bash
# Meridian Phase-II FGMM (zw2) full-year batch -- INDEPENDENT REPLICATION dataset.
#
# Why it matters: Phase-II (2025-2026) does not overlap Phase-I (2010-2022) at all,
# so reproducing the entropy-band results here is a cross-decade / cross-instrument
# (FGM -> FGMM01) replication, not just "more data".
#
# Product: PGCV L1 DAY @ 1 Hz (HDZ), one file per station-day, via the B015 corpus
# link: /data/ssd-stor/YLK/YLK-BUFFER/NSSC-FSF/B015/dataset/data/zw2_fgmm
set -u
R=/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/v8_entropy_dissipation
cd "$R" || exit 1
PY=/data/ssd-stor/fsf/anaconda3/bin/python
YEAR=${1:-2025}
MAXJ=${2:-6}
mkdir -p logs results

STATIONS="OMOHE OYAKS OHUTB OXLHT OCSSL OJIYG ODLZS OKSHI OHLAN OQIMO OTYJY OTANC \
OGAER OYICH OPIXI OENSH OHZXH OBATA OLSCG OSHYA OLJGC ONANA OZHQI OQIZH OLEDO \
OGERM OMZLN ONOAN"

i=0
for st in $STATIONS; do
    (
        $PY code/run_multiband.py --src zw2 --station "$st" --year "$YEAR" \
            > "logs/zw2_${st}_${YEAR}.log" 2>&1 \
            && echo "[done] $st $YEAR" || echo "[FAIL] $st $YEAR"
    ) &
    i=$((i + 1))
    if [ $((i % MAXJ)) -eq 0 ]; then wait; fi
done
wait
echo "[zw2] batch complete for $YEAR"
