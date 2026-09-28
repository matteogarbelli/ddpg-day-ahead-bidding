"""Price data and the stratified train/test split.

Days are indexed 0..n_days-1. A decision taken after observing day d concerns the
delivery (target) day d + 1. The split assigns target days to train or test.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PriceData:
    dates: pd.DatetimeIndex     # one entry per day
    prices: np.ndarray          # (n_days, 24), EUR/MWh

    @property
    def n_days(self) -> int:
        return self.prices.shape[0]


def load_prices(path) -> PriceData:
    """Load the wide price table; a relative path is tried from the working directory, then
    from the repository root."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    table = pd.read_csv(path)
    dates = pd.to_datetime(table["date"])
    prices = table[[f"h{h:02d}" for h in range(1, 25)]].to_numpy(dtype=float)
    if np.isnan(prices).any():
        raise ValueError(f"missing values in {path}")
    return PriceData(pd.DatetimeIndex(dates), prices)


@dataclass
class Split:
    train_targets: np.ndarray   # sorted indices of target days used for training
    test_targets: np.ndarray    # sorted indices of held-out target days


def stratified_split(dates: pd.DatetimeIndex, history_days: int, block_days: int,
                     test_fraction: float, seed: int) -> Split:
    """Hold out whole blocks of consecutive target days, stratified by (year, quarter).

    Target days start at ``history_days`` so that every state has a full price history.
    Consecutive target days are grouped in blocks of ``block_days``; each block belongs to
    the (year, quarter) of its first day. In every stratum, max(1, round(test_fraction * n))
    of its n blocks are drawn at random for the test set.
    """
    targets = np.arange(history_days, len(dates))
    blocks = [targets[i:i + block_days] for i in range(0, len(targets), block_days)]
    strata = {}
    for b, block in enumerate(blocks):
        first = dates[block[0]]
        strata.setdefault((first.year, first.quarter), []).append(b)

    rng = np.random.default_rng(seed)
    test_blocks = set()
    for key in sorted(strata):
        members = strata[key]
        n_test = max(1, int(round(test_fraction * len(members))))
        test_blocks.update(rng.choice(members, size=n_test, replace=False).tolist())

    test = np.concatenate([blocks[b] for b in sorted(test_blocks)])
    train = np.setdiff1d(targets, test)
    return Split(train_targets=train, test_targets=np.sort(test))


def episode_starts(train_targets: np.ndarray, episode_length: int) -> np.ndarray:
    """Decision days d0 whose episode targets d0+1, ..., d0+L are all training days."""
    train = set(train_targets.tolist())
    return np.array([t - 1 for t in train_targets
                     if all(t + j in train for j in range(episode_length))], dtype=int)
