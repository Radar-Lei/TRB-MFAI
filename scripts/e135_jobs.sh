#!/bin/bash
# E135: extended-budget (3600-update) finetuning, flow vs response.
# Anaheim pool starts immediately (10 workers, ~50 min each);
# the Sioux Falls pool starts after the Anaheim pool finishes to keep
# total added load near 20 single-thread processes.
cd "$(dirname "$0")"
PY=../.venv/bin/python
R=results/e135_v109_extbudget
echo "=== e135 anaheim training ($(date)) ==="
for s in 0 1 2 3 4; do
  echo "$PY run_e135_v109_extbudget.py train anaheim $s flow > $R/an_flow_${s}.log 2>&1"
  echo "$PY run_e135_v109_extbudget.py train anaheim $s response > $R/an_response_${s}.log 2>&1"
done | xargs -P 10 -I{} bash -c '{}'
echo "=== e135 sioux falls training ($(date)) ==="
for s in 0 1 2 3 4; do
  echo "$PY run_e135_v109_extbudget.py train sf $s flow > $R/sf_flow_${s}.log 2>&1"
  echo "$PY run_e135_v109_extbudget.py train sf $s response > $R/sf_response_${s}.log 2>&1"
done | xargs -P 10 -I{} bash -c '{}'
echo "=== e135 evaluation ($(date)) ==="
$PY run_e135_v109_extbudget.py eval sf > $R/eval_sf.log 2>&1 &
$PY run_e135_v109_extbudget.py eval anaheim > $R/eval_anaheim.log 2>&1 &
wait
echo "=== e135 done ($(date)) ==="
