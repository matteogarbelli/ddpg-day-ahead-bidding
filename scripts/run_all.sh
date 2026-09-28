#!/usr/bin/env bash
# Train and evaluate every configuration in configs/ for seeds 0, 1, 2 (runs in parallel).
# Usage: bash scripts/run_all.sh [configs ...]
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
configs=("${@:-configs/episode5.json configs/episode7.json configs/episode10.json configs/article_table1.json}")
for config in ${configs[@]}; do
  for seed in 0 1 2; do
    python scripts/train.py --config "$config" --seed "$seed" --eval-every 100 > /dev/null &
  done
done
wait
python scripts/evaluate.py runs/*_seed*
