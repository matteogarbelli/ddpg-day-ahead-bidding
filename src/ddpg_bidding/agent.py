"""Deep Deterministic Policy Gradient (Lillicrap et al., 2015)."""

import copy

import numpy as np
import torch
from torch import nn

from .networks import Actor, Critic


class DDPGAgent:
    def __init__(self, state_dim: int, step_bounds, price_min: float, price_max: float,
                 price_scale: float, hidden_size: int = 64,
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3, weight_decay: float = 1e-4,
                 gamma: float = 0.0, tau: float = 0.01, anchor_index: int = None,
                 price_band: float = 50.0):
        self.n_steps = len(step_bounds)
        total_capacity = float(sum(step_bounds))
        self.gamma = gamma
        self.tau = tau
        self.actor = Actor(state_dim, step_bounds, hidden_size, price_min, price_max,
                           anchor_index, price_band, price_scale)
        self.critic = Critic(state_dim, self.n_steps, hidden_size, price_scale, total_capacity)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    @torch.no_grad()
    def act(self, state: np.ndarray):
        """Deterministic policy mu(s): returns (prices, volumes) as numpy arrays."""
        prices, volumes = self.actor(torch.as_tensor(state).unsqueeze(0))
        return prices[0].numpy().astype(float), volumes[0].numpy().astype(float)

    def update(self, batch):
        """One gradient step on critic and actor, then a soft update of the targets.

        Episodes end by truncation (the auction continues), so every transition is bootstrapped.
        """
        states, actions, rewards, next_states = (torch.as_tensor(x) for x in batch)
        prices, volumes = actions[:, :self.n_steps], actions[:, self.n_steps:]

        with torch.no_grad():
            next_prices, next_volumes = self.actor_target(next_states)
            y = rewards + self.gamma * self.critic_target(next_states, next_prices, next_volumes)
        critic_loss = nn.functional.mse_loss(self.critic(states, prices, volumes), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(states, *self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        return critic_loss.item(), actor_loss.item()

    @torch.no_grad()
    def _soft_update(self, target: nn.Module, source: nn.Module):
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1.0 - self.tau).add_(self.tau * s)
