from pathlib import Path

import numpy as np

from ddpg_bidding.config import load_config
from ddpg_bidding.data import load_prices

ROOT = Path(__file__).resolve().parents[1]


def short_run(seed):
    from ddpg_bidding.train import train

    cfg = load_config(ROOT / "configs" / "episode5.json")
    cfg.train.n_episodes = 20
    cfg.agent.batch_size = 16
    data = load_prices(ROOT / cfg.data.path)
    _, history, evaluations = train(cfg, seed, data=data, eval_every=10, log=lambda *_: None)
    return history, evaluations


def test_training_runs_and_is_reproducible():
    h1, e1 = short_run(seed=3)
    h2, e2 = short_run(seed=3)
    assert len(h1) == 20 and len(e1) == 2
    r1 = np.array([h["reward"] for h in h1])
    assert np.all(np.isfinite(r1))
    assert np.array_equal(r1, np.array([h["reward"] for h in h2]))
    assert np.array_equal([h["critic_loss"] for h in h1[5:]], [h["critic_loss"] for h in h2[5:]])
    assert e1 == e2
