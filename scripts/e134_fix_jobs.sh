#!/bin/bash
# R1 fix rerun: wresponse/kboundary with the correct TSTT marginal weights
# (t0*(1+0.75*(F/cap)^4)), then E137 with e134-identical sampling and
# budget-internal scales.  response/edge/cut runs are unaffected and kept.
cd "$(dirname "$0")"
PY=../.venv/bin/python
R=results/e134_v109_pareto
echo "=== fix stage 1: weighted-objective training ($(date)) ==="
for m in wresponse kboundary; do
  for lam in 0.001 0.003 0.01 0.03 0.1 0.3 1; do
    for s in 0 1 2 3 4; do
      echo "$PY run_e134_v109_pareto.py train --method $m --lam $lam --seed $s > $R/${m}_${lam}_${s}.log 2>&1"
    done
  done
done | xargs -P 16 -I{} bash -c '{}'
echo "=== fix stage 2: select ($(date)) ==="
$PY run_e134_v109_pareto.py select > $R/select.log 2>&1
tail -8 $R/select.log
echo "=== fix stage 3: weighted-objective evals ($(date)) ==="
for m in wresponse kboundary; do
  for lam in 0.001 0.003 0.01 0.03 0.1 0.3 1; do
    echo "$PY run_e134_v109_pareto.py eval --eval-method $m --eval-lam $lam > $R/eval_${m}_${lam}.log 2>&1"
  done
done | xargs -P 14 -I{} bash -c '{}'
echo "=== fix stage 4: e137 rerun (sampling + budget-internal scale) ($(date)) ==="
for q in 6 12 24 48; do
  for s in 0 1 2 3 4; do
    echo "$PY run_e137_v109_labelbudget.py train --per-query $q --seed $s > results/e137_v109_labelbudget/q${q}_${s}.log 2>&1"
  done
done | xargs -P 10 -I{} bash -c '{}'
for q in 6 12 24 48; do
  $PY run_e137_v109_labelbudget.py eval --per-query $q > results/e137_v109_labelbudget/eval_q${q}.log 2>&1
done
echo "=== fix stage 5: collects ($(date)) ==="
$PY run_e134_v109_pareto.py collect > $R/collect.log 2>&1
$PY run_e137_v109_labelbudget.py collect > results/e137_v109_labelbudget/collect.log 2>&1
echo "=== fix rerun done ($(date)) ==="
tail -14 $R/collect.log
tail -8 results/e137_v109_labelbudget/collect.log
