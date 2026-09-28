from pathlib import Path

import numpy as np

from ddpg_bidding.data import episode_starts, load_prices, stratified_split

DATA = Path(__file__).resolve().parents[1] / "data" / "pun_hourly_2017_2020.csv"


def test_data_file():
    data = load_prices(DATA)
    assert data.prices.shape == (1461, 24)
    assert str(data.dates[0].date()) == "2017-01-01" and str(data.dates[-1].date()) == "2020-12-31"
    assert data.prices[0, 0] == 53.30


def test_split_is_disjoint_and_covers_every_stratum():
    data = load_prices(DATA)
    split = stratified_split(data.dates, history_days=7, block_days=14, test_fraction=0.2, seed=0)
    assert len(np.intersect1d(split.train_targets, split.test_targets)) == 0
    assert len(split.train_targets) + len(split.test_targets) == data.n_days - 7
    test_strata = {(d.year, d.quarter) for d in data.dates[split.test_targets]}
    assert test_strata == {(y, q) for y in range(2017, 2021) for q in range(1, 5)}


def test_episodes_only_touch_training_targets():
    data = load_prices(DATA)
    split = stratified_split(data.dates, history_days=7, block_days=14, test_fraction=0.2, seed=0)
    train = set(split.train_targets.tolist())
    for length in (5, 7, 10):
        starts = episode_starts(split.train_targets, length)
        assert len(starts) > 0
        for d0 in starts:
            assert all(d0 + 1 + j in train for j in range(length))
