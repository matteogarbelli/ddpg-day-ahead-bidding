"""DDPG bidding strategy for a single supplier in the Italian day-ahead market.

Reference implementation of Di Persio, Garbelli, Giordano, "Reinforcement learning for
bidding strategy optimization in day-ahead energy market", Energy Economics 149 (2025) 108673.
"""

from .config import Config, load_config

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "__version__"]
