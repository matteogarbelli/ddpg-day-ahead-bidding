"""Clearing of a stepwise offering curve against the realised PUN.

The supplier owns K production modes with unit costs C_k and capacities D_k.
An offering curve has I steps (P_i, V_i): volume V_i is offered at price P_i.
A step is accepted when P_i <= PUN. The accepted volume is produced by the
modes in merit order (cheapest first).
"""

import numpy as np


def accepted_volume(prices, volumes, pun: float) -> float:
    """Q = sum_i V_i 1{P_i <= PUN}."""
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    return float(np.sum(volumes[prices <= pun]))


def dispatch(quantity: float, costs, capacities) -> np.ndarray:
    """Merit-order allocation of ``quantity`` to the production modes.

    Returns q_k, in the order of ``costs``. Raises if the quantity exceeds total capacity
    by more than a relative 1e-6 (float32 rounding of the actor output).
    """
    costs = np.asarray(costs, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    if quantity > capacities.sum() * (1.0 + 1e-6):
        raise ValueError(f"quantity {quantity} exceeds total capacity {capacities.sum()}")
    q = np.zeros_like(capacities)
    remaining = quantity
    for k in np.argsort(costs, kind="stable"):
        q[k] = min(capacities[k], remaining)
        remaining -= q[k]
    return q


def profit(prices, volumes, pun: float, costs, capacities, pricing: str = "bid") -> float:
    """Profit of the offering curve given the realised PUN.

    ``pricing="bid"``: revenue sum_i P_i V_i 1{P_i <= PUN} (Eq. (reward) of the article).
    ``pricing="clearing"``: revenue PUN * Q.
    The production cost is sum_k C_k q_k, with q_k the merit-order dispatch of Q.
    """
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    accepted = prices <= pun
    quantity = float(np.sum(volumes[accepted]))
    if pricing == "bid":
        revenue = float(np.sum(prices[accepted] * volumes[accepted]))
    elif pricing == "clearing":
        revenue = pun * quantity
    else:
        raise ValueError(f"unknown pricing rule: {pricing}")
    cost = float(np.dot(np.asarray(costs, dtype=float), dispatch(quantity, costs, capacities)))
    return revenue - cost


def max_profit(pun: float, costs, capacities) -> float:
    """r_max = sum_k D_k (PUN - C_k)^+ : every profitable mode sold in full at the PUN."""
    costs = np.asarray(costs, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    return float(np.sum(capacities * np.maximum(pun - costs, 0.0)))


def normalized_reward(r: float, r_max: float, floor: float) -> float:
    """r / max(r_max, floor). Equals r / r_max whenever r_max >= floor, and is at most 1."""
    return r / max(r_max, floor)


def project_curve(prices, volumes, price_min: float, price_max: float, bounds):
    """Map a perturbed curve back to the feasible set: sorted prices in [price_min, price_max]
    and step volumes in [0, b_i]."""
    prices = np.clip(np.sort(np.asarray(prices, dtype=float)), price_min, price_max)
    volumes = np.clip(np.asarray(volumes, dtype=float), 0.0, np.asarray(bounds, dtype=float))
    return prices, volumes


def step_bounds(costs, capacities, n_steps: int) -> np.ndarray:
    """Upper bounds b_i on the step volumes, summing to the total capacity.

    With one step per production mode (I = K), step i is bounded by the capacity of the
    i-th cheapest mode; otherwise every step is bounded by total capacity / I.
    """
    costs = np.asarray(costs, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    if n_steps == len(capacities):
        return capacities[np.argsort(costs, kind="stable")]
    return np.full(n_steps, capacities.sum() / n_steps)
