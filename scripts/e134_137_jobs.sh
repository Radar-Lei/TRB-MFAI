#!/bin/bash
# E134 (lambda sweep) + E137 (label budget) on Sioux Falls.
# Stage 1: 175 sweep training runs (16 workers, ~4 min each)
# Stage 2: validation-only lambda selection
# Stage 3: 36 eval jobs (flow + 5 methods x 7 lambdas, 96 paths each, 12 workers)
#          + 20 E137 budget runs (8 workers)
# Stage 4: E137 evals (5 jobs) + collect for both experiments
cd "$(dirname "$0")"
PY=../.venv/bin/python
R=results/e134_v109_pareto
echo "=== e134 stage 1: sweep training ($(date)) ==="
for m in response wresponse kboundary edge cut; do
  for lam in 0.001 0.003 0.01 0.03 0.1 0.3 1; do
    for s in 0 1 2 3 4; do
      echo "$PY run_e134_v109_pareto.py train --method $m --lam $lam --seed $s > $R/${m}_${lam}_${s}.log 2>&1"
    done
  done
done | xargs -P 16 -I{} bash -c '{}'
echo "=== e134 stage 2: select ($(date)) ==="
$PY run_e134_v109_pareto.py select > $R/select.log 2>&1
tail -8 $R/select.log
echo "=== e134 stage 3a: eval all configurations ($(date)) ==="
LAMS="0.001 0.003 0.01 0.03 0.1 0.3 1"
{
  echo "$PY run_e134_v109_pareto.py eval --eval-method flow > $R/eval_flow.log 2>&1"
  for m in response wresponse kboundary edge cut; do
    for lam in $LAMS; do
      echo "$PY run_e134_v109_pareto.py eval --eval-method $m --eval-lam $lam > $R/eval_${m}_${lam}.log 2>&1"
    done
  done
} | xargs -P 12 -I{} bash -c '{}'
echo "=== e134 stage 3b: e137 budget training ($(date)) ==="
for q in 6 12 24 48; do
  for s in 0 1 2 3 4; do
    echo "$PY run_e137_v109_labelbudget.py train --per-query $q --seed $s > results/e137_v109_labelbudget/q${q}_${s}.log 2>&1"
  done
done | xargs -P 10 -I{} bash -c '{}'
echo "=== e134 stage 4: e137 evals + collects ($(date)) ==="
$PY run_e137_v109_labelbudget.py eval --per-query 64 > results/e137_v109_labelbudget/eval_q64.log 2>&1
for q in 6 12 24 48; do
  $PY run_e137_v109_labelbudget.py eval --per-query $q > results/e137_v109_labelbudget/eval_q${q}.log 2>&1
done
$PY run_e134_v109_pareto.py collect > $R/collect.log 2>&1
$PY run_e137_v109_labelbudget.py collect > results/e137_v109_labelbudget/collect.log 2>&1
echo "=== e134/e137 done ($(date)) ==="
tail -12 $R/collect.log
