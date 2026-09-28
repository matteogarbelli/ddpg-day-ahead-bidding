"""Day-ahead bidding environment for a single price-taking supplier.

Time convention. At step t the agent has observed the prices of days d-H+1, ..., d
(H = ``history_days``). It submits an offering curve for the target hour h of day
d + 1, and is paid according to PUN[d + 1, h]. The next state is the price history
of days d-H+2, ..., d+1.

The supplier is a price taker: the next state does not depend on the action.
"""

import numpy as np

from .market import max_profit, normalized_reward, profit


class DayAheadBiddingEnv:
    def __init__(self, prices: np.ndarray, costs, capacities, target_hour: int = 12,
                 history_days: int = 7, pricing: str = "bid", reward_floor: float = 1.0):
        self.prices = np.asarray(prices, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        self.capacities = np.asarray(capacities, dtype=float)
        self.target_hour = target_hour
        self.history_days = history_days
        self.pricing = pricing
        self.floor = reward_floor * self.capacities.sum()
        self.day = None

    def observe(self, day: int) -> np.ndarray:
        """Price history of days day-H+1, ..., day, shape (H, 24)."""
        if day - self.history_days + 1 < 0:
            raise IndexError(f"day {day} has less than {self.history_days} days of history")
        return self.prices[day - self.history_days + 1: day + 1]

    def reset(self, decision_day: int) -> np.ndarray:
        self.day = decision_day
        return self.observe(self.day)

    def step(self, bid_prices, bid_volumes):
        """Clear the offering curve against PUN[day + 1, target_hour] and advance one day.

        Returns (next_state, normalised_reward, info).
        """
        pun = float(self.prices[self.day + 1, self.target_hour])
        r = profit(bid_prices, bid_volumes, pun, self.costs, self.capacities, self.pricing)
        r_max = max_profit(pun, self.costs, self.capacities)
        reward = normalized_reward(r, r_max, self.floor)
        self.day += 1
        info = {"pun": pun, "profit": r, "max_profit": r_max}
        return self.observe(self.day), reward, info
