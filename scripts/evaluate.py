"""Evaluate trained runs on the held-out target days, next to the persistence baseline.

Usage::

    python scripts/evaluate.py runs/episode5_seed0 [runs/episode5_seed1 ...]

For each run, prints and saves (evaluation.json) the mean normalised reward, mean profit
and mean r_max of the deterministic policy on the train and test target days. With several
runs of one configuration, also prints the mean and standard deviation over the runs of the
test normalised reward.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ddpg_bidding.evaluate import evaluate_policy, persistence_policy
from ddpg_bidding.train import load_run


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained runs.")
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    test_rewards = {}   # (config name, policy) -> test normalised reward of each run
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
            test_rewards.setdefault((cfg.name, name), []).append(r["test"]["normalized_reward"])

    if any(len(x) > 1 for x in test_rewards.values()):
        print("\ntest normalised reward, mean ± standard deviation over runs")
        for (config, policy), x in test_rewards.items():
            sd = np.std(x, ddof=1) if len(x) > 1 else float("nan")
            print(f"  {config:12s} {policy:12s} {np.mean(x):.3f} ± {sd:.3f}   (runs: {len(x)})")


if __name__ == "__main__":
    main()
