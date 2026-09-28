"""Run configuration, loaded from the JSON manifests in ``configs/``."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    path: str = "data/pun_hourly_2017_2020.csv"
    target_hour: int = 12          # 0-based column of the delivery hour (12 -> 13:00-14:00)
    history_days: int = 7          # days of prices in the state
    block_days: int = 14           # length of the blocks used by the stratified split
    test_fraction: float = 0.2     # share of blocks held out in each (year, quarter) stratum
    split_seed: int = 0


@dataclass
class MarketConfig:
    costs: list = field(default_factory=lambda: [10.0, 30.0, 60.0])          # C_k, EUR/MWh
    capacities: list = field(default_factory=lambda: [30.0, 200.0, 800.0])   # D_k, MWh
    n_steps: int = 3               # I, steps of the offering curve
    price_min: float = 0.0         # EUR/MWh
    price_max: float = 200.0       # EUR/MWh
    pricing: str = "bid"           # "bid": accepted step i earns P_i; "clearing": it earns the PUN
    reward_floor: float = 1.0      # EUR/MWh; r_max is bounded below by reward_floor * sum(D_k)
    price_scale: float = 100.0     # EUR/MWh, divides prices and costs in the network inputs
    price_anchor: bool = True      # centre the actor's prices on the last observed target-hour PUN
    price_band: float = 50.0       # EUR/MWh, half-width of the anchored price range


@dataclass
class AgentConfig:
    hidden_size: int = 64
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    weight_decay: float = 1e-4     # L2 regularisation of actor and critic
    gamma: float = 0.0
    tau: float = 0.01
    buffer_size: int = 50_000
    batch_size: int = 64
    updates_per_step: int = 20     # gradient steps per environment step


@dataclass
class NoiseConfig:
    theta: float = 0.15
    mu: float = 1.0
    sigma_start: float = 10.0
    sigma_end: float = 1.0
    dt: float = 1.0


@dataclass
class TrainConfig:
    n_episodes: int = 1000
    episode_length: int = 5        # days per episode


@dataclass
class Config:
    name: str = "episode5"
    data: DataConfig = field(default_factory=DataConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path) -> Config:
    raw = json.loads(Path(path).read_text())
    return Config(
        name=raw.get("name", Path(path).stem),
        data=DataConfig(**raw.get("data", {})),
        market=MarketConfig(**raw.get("market", {})),
        agent=AgentConfig(**raw.get("agent", {})),
        noise=NoiseConfig(**raw.get("noise", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
