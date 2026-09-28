#!/usr/bin/env bash
# Train every configuration for seeds 0, 1, 2 (runs in parallel), then evaluate the runs.
# Usage: bash scripts/run_all.sh [config ...]    (default: configs/*.json)
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ $# -gt 0 ] || set -- configs/*.json
pids=()
runs=()
for config in "$@"; do
  name=$(basename "$config" .json)   # equals the "name" field, which names the run directory
  for seed in 0 1 2; do
    python scripts/train.py --config "$config" --seed "$seed" --eval-every 100 > /dev/null &
    pids+=($!)
    runs+=("runs/${name}_seed${seed}")
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
python scripts/evaluate.py "${runs[@]}"
