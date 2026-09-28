import numpy as np
import pytest

from ddpg_bidding.env import DayAheadBiddingEnv
from ddpg_bidding.market import max_profit, profit

COSTS = [10.0, 30.0, 60.0]
CAPS = [30.0, 200.0, 800.0]


def synthetic_prices(n_days=30):
    # price of day d, hour h = 100 d + h: every entry identifies its day and hour
    return 100.0 * np.arange(n_days)[:, None] + np.arange(24)[None, :]


def test_state_is_the_last_seven_days():
    env = DayAheadBiddingEnv(synthetic_prices(), COSTS, CAPS, target_hour=12, history_days=7)
    state = env.reset(decision_day=10)
    assert state.shape == (7, 24)
    assert np.array_equal(state[:, 0] // 100, np.arange(4, 11))


def test_reward_uses_next_day_target_hour_and_state_advances():
    prices = np.full((30, 24), 40.0)
    prices[11, 12] = 55.0                       # delivery day 11, hour 13:00-14:00
    env = DayAheadBiddingEnv(prices, COSTS, CAPS, target_hour=12, history_days=7, reward_floor=1.0)
    env.reset(decision_day=10)
    bid_p, bid_v = [20.0, 40.0, 55.0], [100.0, 100.0, 100.0]
    next_state, reward, info = env.step(bid_p, bid_v)
    assert info["pun"] == 55.0
    expected = profit(bid_p, bid_v, 55.0, COSTS, CAPS) / max_profit(55.0, COSTS, CAPS)
    assert reward == pytest.approx(expected)
    assert np.array_equal(next_state, prices[5:12])


def test_state_never_contains_the_delivery_day():
    prices = synthetic_prices()
    env = DayAheadBiddingEnv(prices, COSTS, CAPS, target_hour=12, history_days=7)
    state = env.reset(decision_day=10)
    for _ in range(5):
        delivery_day = env.day + 1
        assert state.max() < 100.0 * delivery_day
        state, _, info = env.step([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        assert info["pun"] == 100.0 * delivery_day + 12


def test_insufficient_history_raises():
    env = DayAheadBiddingEnv(synthetic_prices(), COSTS, CAPS, history_days=7)
    with pytest.raises(IndexError):
        env.reset(decision_day=5)
