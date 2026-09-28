"""Replay buffer of transitions (s, a, r, s')."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward, next_state):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]
