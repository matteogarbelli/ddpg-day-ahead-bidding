import numpy as np
import torch

from ddpg_bidding.networks import Actor, Critic, encode_state


def test_actor_output_is_a_feasible_curve():
    torch.manual_seed(0)
    actor = Actor(state_dim=174, step_bounds=[30.0, 200.0, 800.0], hidden=64, price_min=0.0,
                  price_max=200.0)
    # large random weights push the sigmoid heads towards their extremes
    for p in actor.parameters():
        torch.nn.init.normal_(p, std=3.0)
    prices, volumes = actor(torch.randn(512, 174))
    assert prices.shape == volumes.shape == (512, 3)
    assert torch.all(prices[:, 1:] >= prices[:, :-1])
    assert torch.all(prices >= 0.0) and torch.all(prices <= 200.0)
    assert torch.all(volumes >= 0.0)
    assert torch.all(volumes <= torch.tensor([30.0, 200.0, 800.0]))


def test_critic_returns_one_value_per_sample():
    critic = Critic(state_dim=174, n_steps=3, hidden=64, price_scale=100.0, total_capacity=1030.0)
    q = critic(torch.randn(8, 174), torch.rand(8, 3) * 100, torch.rand(8, 3) * 300)
    assert q.shape == (8,)


def test_encode_state_is_day_major():
    history = 100.0 * np.arange(7)[:, None] + np.arange(24)[None, :]
    x = encode_state(history, [10.0, 30.0, 60.0], [30.0, 200.0, 800.0], price_scale=100.0)
    assert x.shape == (7 * 24 + 6,)
    assert x[24] == 1.0                           # day 1, hour 0
    assert np.isclose(x[-3:].sum(), 1.0)          # capacities scaled by their total


def test_anchored_prices_are_centred_on_the_reference():
    torch.manual_seed(0)
    anchor = 6 * 24 + 12
    actor = Actor(state_dim=174, step_bounds=[30.0, 200.0, 800.0], hidden=64, price_min=0.0,
                  price_max=200.0, anchor_index=anchor, band=50.0, price_scale=100.0)
    state = torch.zeros(4, 174)
    state[:, anchor] = torch.tensor([0.2, 0.5, 0.8, 1.9])      # reference prices 20, 50, 80, 190
    prices, _ = actor(state)
    # small initial output weights: every step starts close to the reference price
    assert torch.allclose(prices, torch.tensor([[20.0], [50.0], [80.0], [190.0]]).expand(4, 3), atol=1.0)
    assert torch.all(prices <= 200.0)
