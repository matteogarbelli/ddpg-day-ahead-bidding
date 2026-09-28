"""Plot training curves and an offering curve of a trained run.

Usage::

    python scripts/plot.py runs/episode5_seed0 [--day 2020-06-15]

Writes to <run>/figures/: reward.png (normalised reward per episode and its 10-episode
moving average; the y-axis starts at the 1st percentile of the episode rewards), losses.png (critic and actor loss per episode) and offering_curve.png
(the policy's curve for one test day, with the realised PUN).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ddpg_bidding.train import load_run


def plot_reward(metrics: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics["episode"], metrics["reward"], lw=0.6, alpha=0.5, label="episode")
    ax.plot(metrics["episode"], metrics["reward"].rolling(10).mean(), lw=1.8,
            label="10-episode moving average")
    low, high = np.nanpercentile(metrics["reward"], 1), np.nanmax(metrics["reward"])
    margin = 0.05 * (high - low)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("episode")
    ax.set_ylabel("normalised reward")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_losses(metrics: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    ax1.plot(metrics["episode"], metrics["critic_loss"], lw=0.8, color="C2")
    ax1.set_ylabel("critic loss")
    ax1.set_yscale("log")
    ax2.plot(metrics["episode"], metrics["actor_loss"], lw=0.8, color="C1")
    ax2.set_ylabel("actor loss")
    ax2.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_offering_curve(cfg, setup, target: int, path: Path):
    history = setup.env.observe(target - 1)
    prices, volumes = setup.greedy_policy(cfg)(history)
    pun = setup.env.prices[target, cfg.data.target_hour]
    cumulative = np.concatenate([[0.0], np.cumsum(volumes)])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(cumulative, np.concatenate([prices, prices[-1:]]), where="post", marker="o",
            label="offering curve")
    ax.axhline(pun, color="C3", ls="--", label="realised PUN")
    ax.set_xlabel("cumulative volume (MWh)")
    ax.set_ylabel("price (EUR/MWh)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a trained run.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--day", default=None, help="test delivery day (YYYY-MM-DD); default: last test day")
    args = parser.parse_args()

    cfg, setup = load_run(args.run)
    out = args.run / "figures"
    out.mkdir(exist_ok=True)
    metrics = pd.read_csv(args.run / "metrics.csv")
    plot_reward(metrics, out / "reward.png")
    plot_losses(metrics, out / "losses.png")

    test = setup.split.test_targets
    target = int(test[-1]) if args.day is None else int(setup.data.dates.get_loc(pd.Timestamp(args.day)))
    if target not in set(test.tolist()):
        raise SystemExit(f"{setup.data.dates[target].date()} is not a test day")
    plot_offering_curve(cfg, setup, target, out / "offering_curve.png")
    print(f"figures written to {out} (offering curve for {setup.data.dates[target].date()})")


if __name__ == "__main__":
    main()
