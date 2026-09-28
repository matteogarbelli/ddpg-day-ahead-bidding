"""Evaluation of bidding policies on a set of target days."""

import numpy as np

from .env import DayAheadBiddingEnv


def evaluate_policy(policy, env: DayAheadBiddingEnv, targets: np.ndarray) -> dict:
    """Apply ``policy(history) -> (prices, volumes)`` once for every target day.

    Returns means over the target days of the normalised reward, the profit and r_max (EUR),
    the offered and accepted volume (MWh), and the ratio of total profit to total r_max.
    """
    rewards, profits, max_profits, offered, accepted = [], [], [], [], []
    for target in targets:
        history = env.reset(int(target) - 1)
        prices, volumes = policy(history)
        _, reward, info = env.step(prices, volumes)
        rewards.append(reward)
        profits.append(info["profit"])
        max_profits.append(info["max_profit"])
        offered.append(float(np.sum(volumes)))
        accepted.append(float(np.sum(np.asarray(volumes)[np.asarray(prices) <= info["pun"]])))
    return {
        "normalized_reward": float(np.mean(rewards)),
        "profit_ratio": float(np.sum(profits) / np.sum(max_profits)),
        "profit": float(np.mean(profits)),
        "max_profit": float(np.mean(max_profits)),
        "offered_volume": float(np.mean(offered)),
        "accepted_volume": float(np.mean(accepted)),
        "n_days": int(len(targets)),
    }


def persistence_policy(costs, capacities, target_hour: int):
    """Reference strategy: forecast the PUN by its value on the last observed day, and offer
    each mode's full capacity at max(C_k, forecast). Requires one curve step per mode."""
    costs = np.asarray(costs, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    order = np.argsort(costs, kind="stable")

    def policy(history):
        forecast = history[-1, target_hour]
        return np.maximum(costs[order], forecast), capacities[order]

    return policy
