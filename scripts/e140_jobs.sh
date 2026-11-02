#!/bin/bash
cd "$(dirname "$0")"
mkdir -p results/e140_v110_resolution
PY=../.venv/bin/python
{
for m in flow response; do for s in 0 1 2 3 4; do
  echo "$PY run_e140_v110_resolution.py sf --method $m --seed $s > results/e140_v110_resolution/sf_${m}_${s}.log 2>&1"
done; done
for m in flow response; do for s in 0 1 2 3 4; do
  echo "$PY run_e140_v110_resolution.py an --method $m --seed $s > results/e140_v110_resolution/an_${m}_${s}.log 2>&1"
done; done
} | xargs -P 10 -I{} bash -c '{}'
echo "=== shards done $(date) ==="
$PY run_e140_v110_resolution.py collect > results/e140_v110_resolution/collect.log 2>&1
tail -80 results/e140_v110_resolution/collect.log
echo "=== e140 done $(date) ==="
