"""Evaluate trained runs on the held-out target days, next to the persistence baseline.

Usage::

    python scripts/evaluate.py runs/episode5_seed0 [runs/episode5_seed1 ...]

For each run, prints and saves (evaluation.json) the mean normalised reward, mean profit
and mean r_max of the deterministic policy on the train and test target days.
"""

import argparse
import json
from pathlib import Path

from ddpg_bidding.evaluate import evaluate_policy, persistence_policy
from ddpg_bidding.train import load_run


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained runs.")
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    for run_dir in args.runs:
        cfg, setup = load_run(run_dir)
        policies = {"ddpg": setup.greedy_policy(cfg)}
        if cfg.market.n_steps == len(cfg.market.costs):
            policies["persistence"] = persistence_policy(cfg.market.costs, cfg.market.capacities,
                                                         cfg.data.target_hour)
        result = {}
        for name, policy in policies.items():
            result[name] = {
                "train": evaluate_policy(policy, setup.env, setup.split.train_targets),
                "test": evaluate_policy(policy, setup.env, setup.split.test_targets),
            }
        (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(run_dir)
        for name, r in result.items():
            print(f"  {name:12s} normalised reward  train {r['train']['normalized_reward']:+.3f}"
                  f"  test {r['test']['normalized_reward']:+.3f}   (test days: {r['test']['n_days']})")


if __name__ == "__main__":
    main()
