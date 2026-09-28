"""Actor and critic networks.

The actor maps a state to an offering curve that is feasible by construction:
non-decreasing prices in (price_min, price_max) and step volumes V_i in (0, b_i),
where the step bounds b_i sum to the total capacity (see ``market.step_bounds``).
"""

import numpy as np
import torch
from torch import nn


def encode_state(history: np.ndarray, costs, capacities, price_scale: float) -> np.ndarray:
    """Network input: price history (day-major) and costs scaled by ``price_scale``,
    capacities scaled by their total."""
    capacities = np.asarray(capacities, dtype=float)
    return np.concatenate([
        np.asarray(history, dtype=float).reshape(-1) / price_scale,
        np.asarray(costs, dtype=float) / price_scale,
        capacities / capacities.sum(),
    ]).astype(np.float32)


def _mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_out),
    )
    # Small output weights, as in Lillicrap et al. (2015).
    nn.init.uniform_(net[-1].weight, -3e-3, 3e-3)
    nn.init.uniform_(net[-1].bias, -3e-3, 3e-3)
    return net


class Actor(nn.Module):
    """Offering curve mu(s) = (P, V).

    Prices, without an anchor: P_i = price_min + (price_max - price_min) sigmoid(z_i).
    Prices, with an anchor:    P_i = clip(p_ref + band tanh(z_i), price_min, price_max), where
    p_ref = state[anchor_index] * price_scale is a price read from the state.
    Prices are sorted increasingly. Volumes: V_i = b_i sigmoid(y_i).
    """

    def __init__(self, state_dim: int, step_bounds, hidden: int, price_min: float, price_max: float,
                 anchor_index: int = None, band: float = 50.0, price_scale: float = 100.0):
        super().__init__()
        self.n_steps = len(step_bounds)
        self.price_min = price_min
        self.price_max = price_max
        self.anchor_index = anchor_index
        self.band = band
        self.price_scale = price_scale
        self.register_buffer("step_bounds", torch.as_tensor(step_bounds, dtype=torch.float32))
        self.body = _mlp(state_dim, hidden, 2 * self.n_steps)

    def forward(self, state: torch.Tensor):
        """Return (prices, volumes), each of shape (batch, n_steps), in EUR/MWh and MWh."""
        z, y = self.body(state).split(self.n_steps, dim=-1)
        if self.anchor_index is None:
            prices = self.price_min + (self.price_max - self.price_min) * torch.sigmoid(z)
        else:
            reference = state[:, self.anchor_index:self.anchor_index + 1] * self.price_scale
            prices = torch.clamp(reference + self.band * torch.tanh(z), self.price_min, self.price_max)
        prices, _ = torch.sort(prices, dim=-1)
        volumes = self.step_bounds * torch.sigmoid(y)
        return prices, volumes


class Critic(nn.Module):
    def __init__(self, state_dim: int, n_steps: int, hidden: int, price_scale: float,
                 total_capacity: float):
        super().__init__()
        self.price_scale = price_scale
        self.total_capacity = total_capacity
        self.body = _mlp(state_dim + 2 * n_steps, hidden, 1)

    def forward(self, state: torch.Tensor, prices: torch.Tensor, volumes: torch.Tensor) -> torch.Tensor:
        """Q(s, a), shape (batch,)."""
        x = torch.cat([state, prices / self.price_scale, volumes / self.total_capacity], dim=-1)
        return self.body(x).squeeze(-1)
