import numpy as np
import pytest

from ddpg_bidding.market import (
    accepted_volume,
    dispatch,
    max_profit,
    normalized_reward,
    profit,
    project_curve,
    step_bounds,
)

COSTS = [10.0, 30.0, 60.0]
CAPS = [30.0, 200.0, 800.0]


def test_hand_computed_case():
    prices, volumes, pun = [20.0, 40.0, 55.0], [100.0, 100.0, 100.0], 50.0
    assert accepted_volume(prices, volumes, pun) == 200.0
    assert np.allclose(dispatch(200.0, COSTS, CAPS), [30.0, 170.0, 0.0])
    # revenue 20*100 + 40*100 = 6000; cost 30*10 + 170*30 = 5400
    assert profit(prices, volumes, pun, COSTS, CAPS, "bid") == pytest.approx(600.0)
    # revenue 50*200 = 10000
    assert profit(prices, volumes, pun, COSTS, CAPS, "clearing") == pytest.approx(4600.0)
    # 30*(50-10) + 200*(50-30)
    assert max_profit(pun, COSTS, CAPS) == pytest.approx(5200.0)


def test_price_equal_to_pun_is_accepted():
    assert accepted_volume([50.0], [10.0], 50.0) == 10.0
    assert accepted_volume([50.01], [10.0], 50.0) == 0.0


def test_dispatch_is_merit_order_for_unsorted_costs():
    q = dispatch(100.0, [60.0, 10.0, 30.0], [800.0, 30.0, 200.0])
    assert np.allclose(q, [0.0, 30.0, 70.0])


def test_dispatch_rejects_excess_volume():
    with pytest.raises(ValueError):
        dispatch(1031.0, COSTS, CAPS)


def test_profit_never_exceeds_max_profit():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        pun = rng.uniform(0.0, 150.0)
        prices = np.sort(rng.uniform(0.0, 200.0, 3))
        volumes = rng.dirichlet(np.ones(4))[:3] * sum(CAPS)
        r_max = max_profit(pun, COSTS, CAPS)
        for pricing in ("bid", "clearing"):
            assert profit(prices, volumes, pun, COSTS, CAPS, pricing) <= r_max + 1e-6


def test_optimal_curve_reaches_normalised_reward_one():
    pun = 50.0
    r = profit([pun, pun], [30.0, 200.0], pun, COSTS, CAPS, "bid")
    assert normalized_reward(r, max_profit(pun, COSTS, CAPS), floor=1030.0) == pytest.approx(1.0)


def test_floor_applies_when_no_profit_is_possible():
    pun = 5.0
    assert max_profit(pun, COSTS, CAPS) == 0.0
    r = profit([4.0], [30.0], pun, COSTS, CAPS, "bid")   # 30 * (4 - 10)
    assert normalized_reward(r, 0.0, floor=1030.0) == pytest.approx(-180.0 / 1030.0)


def test_project_curve_restores_feasibility():
    bounds = step_bounds(COSTS, CAPS, 3)
    prices, volumes = project_curve([120.0, -5.0, 250.0], [-3.0, 900.0, 600.0], 0.0, 200.0, bounds)
    assert np.allclose(prices, [0.0, 120.0, 200.0])
    assert np.allclose(volumes, [0.0, 200.0, 600.0])


def test_step_bounds():
    assert np.allclose(step_bounds([60.0, 10.0, 30.0], [800.0, 30.0, 200.0], 3), [30.0, 200.0, 800.0])
    assert np.allclose(step_bounds(COSTS, CAPS, 5), [206.0] * 5)
