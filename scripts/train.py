"""Train a DDPG bidding agent from a JSON configuration.

Usage::

    python scripts/train.py --config configs/episode5.json --seed 0

Writes to runs/<config name>_seed<seed>/: the resolved configuration, per-episode
training metrics, periodic test evaluations and the actor and critic weights.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ddpg_bidding.config import load_config
from ddpg_bidding.train import train


def main():
    parser = argparse.ArgumentParser(description="Train a DDPG bidding agent.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--episodes", type=int, default=None, help="override train.n_episodes")
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(1)  # same arithmetic as scripts/run_all.sh

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg.train.n_episodes = args.episodes
    run_dir = args.out / f"{cfg.name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    setup, history, evaluations = train(cfg, args.seed, eval_every=args.eval_every)

    (run_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "seed": args.seed}, indent=2))
    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(evaluations).to_csv(run_dir / "test_evaluations.csv", index=False)
    torch.save(setup.agent.actor.state_dict(), run_dir / "actor.pt")
    torch.save(setup.agent.critic.state_dict(), run_dir / "critic.pt")
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
