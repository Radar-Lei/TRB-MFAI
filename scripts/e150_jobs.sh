#!/bin/bash
# E150 (v112): exact-parameterization rows under the scaled pretraining loss.
# Stage 1: 5 pretraining runs (parallel)
# Stage 2: 10 continuations (parallel)
# Stage 3: evaluation (single process)
# Stage 4: collect (merge with frozen e146 rows)
cd "$(dirname "$0")"
PY=../.venv/bin/python
R=results/e150_v112_exactscaled
echo "=== e150 stage 1: pretrain x5 ($(date)) ==="
for s in 0 1 2 3 4; do
  echo "$PY run_e150_v112_exactscaled.py pretrain --seed $s > $R/pre_${s}.log 2>&1"
done | xargs -P 5 -I{} bash -c '{}'
echo "=== e150 stage 2: cont x10 ($(date)) ==="
for m in flow response; do
  for s in 0 1 2 3 4; do
    echo "$PY run_e150_v112_exactscaled.py cont --seed $s --method $m > $R/cont_${m}_${s}.log 2>&1"
  done
done | xargs -P 10 -I{} bash -c '{}'
echo "=== e150 stage 3: eval ($(date)) ==="
$PY run_e150_v112_exactscaled.py eval > $R/eval.log 2>&1
echo "=== e150 stage 4: collect ($(date)) ==="
$PY run_e150_v112_exactscaled.py collect > $R/collect.log 2>&1
tail -40 $R/collect.log
echo "=== e150 done ($(date)) ==="
