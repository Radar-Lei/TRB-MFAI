#!/bin/bash
# E139 (v110): normalized weighted-response objective.
# Stage 1: 35 training runs (16 workers)
# Stage 2: validation-only lambda selection
# Stage 3: 7 eval jobs (one per lambda, 96 paths each, 7 workers)
# Stage 4: collect
cd "$(dirname "$0")"
PY=../.venv/bin/python
R=results/e139_v110_wnorm
echo "=== e139 stage 1: training ($(date)) ==="
for lam in 0.001 0.003 0.01 0.03 0.1 0.3 1; do
  for s in 0 1 2 3 4; do
    echo "$PY run_e139_v110_wnorm.py train --lam $lam --seed $s > $R/wresponse_${lam}_${s}.log 2>&1"
  done
done | xargs -P 16 -I{} bash -c '{}'
echo "=== e139 stage 2: select ($(date)) ==="
$PY run_e139_v110_wnorm.py select > $R/select.log 2>&1
tail -5 $R/select.log
echo "=== e139 stage 3: eval ($(date)) ==="
for lam in 0.001 0.003 0.01 0.03 0.1 0.3 1; do
  echo "$PY run_e139_v110_wnorm.py eval --lam $lam > $R/eval_wresponse_${lam}.log 2>&1"
done | xargs -P 7 -I{} bash -c '{}'
echo "=== e139 stage 4: collect ($(date)) ==="
$PY run_e139_v110_wnorm.py collect > $R/collect.log 2>&1
tail -30 $R/collect.log
echo "=== e139 done ($(date)) ==="
