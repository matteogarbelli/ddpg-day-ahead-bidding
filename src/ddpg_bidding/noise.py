"""Ornstein-Uhlenbeck exploration noise.

dX_t = -theta (X_t - mu) dt + sigma dW_t, discretised by Euler-Maruyama:
X_{t+dt} = X_t - theta (X_t - mu) dt + sigma sqrt(dt) xi_t,  xi_t ~ N(0, I).
"""

import numpy as np


class OUNoise:
    def __init__(self, dim: int, theta: float, mu: float, dt: float, rng: np.random.Generator):
        self.dim = dim
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.rng = rng
        self.reset()

    def reset(self):
        self.state = np.full(self.dim, self.mu, dtype=float)

    def sample(self, sigma: float) -> np.ndarray:
        xi = self.rng.standard_normal(self.dim)
        self.state = self.state - self.theta * (self.state - self.mu) * self.dt + sigma * np.sqrt(self.dt) * xi
        return self.state.copy()


def linear_sigma(episode: int, n_episodes: int, sigma_start: float, sigma_end: float) -> float:
    """sigma decreasing linearly from sigma_start (first episode) to sigma_end (last episode)."""
    if n_episodes <= 1:
        return sigma_end
    frac = episode / (n_episodes - 1)
    return sigma_start + frac * (sigma_end - sigma_start)

