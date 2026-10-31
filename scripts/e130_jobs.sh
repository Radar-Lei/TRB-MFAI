#!/bin/bash
cd "$(dirname "$0")"
PY=../.venv/bin/python
echo "=== stage A: pretrain x5 ==="
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 $PY run_e130_v108_anaheim_train.py pretrain --seed $s > results/e130_v108_anaheim_train/pre_$s.log 2>&1 &
done
wait
echo "=== stage B: finetune flow+response x5 ==="
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 $PY run_e130_v108_anaheim_train.py finetune --seed $s --method flow > results/e130_v108_anaheim_train/flow_$s.log 2>&1 &
  OMP_NUM_THREADS=1 $PY run_e130_v108_anaheim_train.py finetune --seed $s --method response > results/e130_v108_anaheim_train/response_$s.log 2>&1 &
done
wait
echo "=== e130 done ==="
