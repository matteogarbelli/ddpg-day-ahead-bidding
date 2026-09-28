"""DDPG training loop."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .agent import DDPGAgent
from .config import Config, load_config
from .data import PriceData, Split, episode_starts, load_prices, stratified_split
from .env import DayAheadBiddingEnv
from .evaluate import evaluate_policy
from .market import project_curve, step_bounds
from .networks import encode_state
from .noise import OUNoise, linear_sigma
from .replay import ReplayBuffer


@dataclass
class Setup:
    data: PriceData
    split: Split
    env: DayAheadBiddingEnv
    agent: DDPGAgent
    state_dim: int

    def encode(self, history: np.ndarray, cfg: Config) -> np.ndarray:
        return encode_state(history, cfg.market.costs, cfg.market.capacities, cfg.market.price_scale)

    def greedy_policy(self, cfg: Config):
        return lambda history: self.agent.act(self.encode(history, cfg))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(cfg: Config, seed: int, data: PriceData = None) -> Setup:
    """Data, split, environment and a freshly initialised agent for ``cfg``."""
    seed_everything(seed)
    data = data if data is not None else load_prices(cfg.data.path)
    split = stratified_split(data.dates, cfg.data.history_days, cfg.data.block_days,
                             cfg.data.test_fraction, cfg.data.split_seed)
    m = cfg.market
    env = DayAheadBiddingEnv(data.prices, m.costs, m.capacities, cfg.data.target_hour,
                             cfg.data.history_days, m.pricing, m.reward_floor)
    state_dim = cfg.data.history_days * 24 + 2 * len(m.costs)
    a = cfg.agent
    # position of PUN[d, target_hour] in the encoded state (last day of the history)
    anchor = (cfg.data.history_days - 1) * 24 + cfg.data.target_hour if m.price_anchor else None
    agent = DDPGAgent(state_dim, step_bounds(m.costs, m.capacities, m.n_steps), m.price_min,
                      m.price_max, m.price_scale, a.hidden_size, a.actor_lr, a.critic_lr,
                      a.weight_decay, a.gamma, a.tau, anchor, m.price_band)
    return Setup(data, split, env, agent, state_dim)


def train(cfg: Config, seed: int, data: PriceData = None, eval_every: int = 50, log=print):
    """Train one agent. Returns (setup, per-episode metrics, periodic test evaluations)."""
    setup = build(cfg, seed, data)
    env, agent = setup.env, setup.agent
    m, t = cfg.market, cfg.train
    rng = np.random.default_rng(seed)
    bounds = step_bounds(m.costs, m.capacities, m.n_steps)
    starts = episode_starts(setup.split.train_targets, t.episode_length)
    buffer = ReplayBuffer(cfg.agent.buffer_size, setup.state_dim, 2 * m.n_steps)
    noise = OUNoise(2 * m.n_steps, cfg.noise.theta, cfg.noise.mu, cfg.noise.dt, rng)

    history, evaluations = [], []
    for episode in range(t.n_episodes):
        sigma = linear_sigma(episode, t.n_episodes, cfg.noise.sigma_start, cfg.noise.sigma_end)
        state = setup.encode(env.reset(int(rng.choice(starts))), cfg)
        noise.reset()
        rewards, critic_losses, actor_losses = [], [], []
        for _ in range(t.episode_length):
            prices, volumes = agent.act(state)
            x = noise.sample(sigma)
            prices, volumes = project_curve(prices + x[:m.n_steps], volumes + x[m.n_steps:],
                                            m.price_min, m.price_max, bounds)
            next_history, reward, _ = env.step(prices, volumes)
            next_state = setup.encode(next_history, cfg)
            buffer.add(state, np.concatenate([prices, volumes]), reward, next_state)
            if len(buffer) >= cfg.agent.batch_size:
                for _ in range(cfg.agent.updates_per_step):
                    c_loss, a_loss = agent.update(buffer.sample(cfg.agent.batch_size, rng))
                    critic_losses.append(c_loss)
                    actor_losses.append(a_loss)
            rewards.append(reward)
            state = next_state

        history.append({
            "episode": episode,
            "reward": float(np.mean(rewards)),
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
            "sigma": sigma,
        })
        if (episode + 1) % eval_every == 0 or episode + 1 == t.n_episodes:
            result = evaluate_policy(setup.greedy_policy(cfg), env, setup.split.test_targets)
            evaluations.append({"episode": episode, **result})
            log(f"episode {episode + 1:5d}  train reward {history[-1]['reward']:+.3f}  "
                f"test reward {result['normalized_reward']:+.3f}")
    return setup, history, evaluations


def load_run(run_dir) -> tuple:
    """Rebuild the configuration, data split and trained actor of a saved run."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    seed = json.loads((run_dir / "config.json").read_text())["seed"]
    setup = build(cfg, seed)
    setup.agent.actor.load_state_dict(torch.load(run_dir / "actor.pt", weights_only=True))
    return cfg, setup
