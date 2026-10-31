#!/bin/bash
# e129 parallel data generation: flows -> centers -> paths -> collect
cd "$(dirname "$0")"
PY=../.venv/bin/python
NW=${NW:-12}
echo "=== flows ($NW workers) ==="
for i in $(seq 0 $((NW-1))); do
  OMP_NUM_THREADS=1 $PY run_e129_v108_anaheim_data.py solve --task flows --worker $i --nworkers $NW > results/e129_v108_anaheim_data/flows_w$i.log 2>&1 &
done
wait
echo "=== centers ($NW workers) ==="
for i in $(seq 0 $((NW-1))); do
  OMP_NUM_THREADS=1 $PY run_e129_v108_anaheim_data.py solve --task centers --worker $i --nworkers $NW > results/e129_v108_anaheim_data/centers_w$i.log 2>&1 &
done
wait
echo "=== paths ($NW workers) ==="
for i in $(seq 0 $((NW-1))); do
  OMP_NUM_THREADS=1 $PY run_e129_v108_anaheim_data.py solve --task paths --worker $i --nworkers $NW > results/e129_v108_anaheim_data/paths_w$i.log 2>&1 &
done
wait
echo "=== collect ==="
OMP_NUM_THREADS=1 $PY run_e129_v108_anaheim_data.py collect 2>&1 | tail -20
echo "=== e129 done ==="
