# DDPG bidding in the day-ahead electricity market

Reference implementation of the single-agent model of

> L. Di Persio, M. Garbelli, L. M. Giordano, *Reinforcement learning for bidding strategy
> optimization in day-ahead energy market*, Energy Economics 149 (2025) 108673.
> [doi:10.1016/j.eneco.2025.108673](https://doi.org/10.1016/j.eneco.2025.108673)

A supplier with $K$ production modes submits, every day, a stepwise offering curve for one
delivery hour of the next day in the Italian day-ahead market (MGP). The curve is chosen by a
Deep Deterministic Policy Gradient (DDPG) agent from the last seven days of hourly national
prices (PUN). The code follows Sections 2–4 of the article; the settings that differ from its
Table 1 are listed under [Settings](#settings), and the numbers it produces are not those of
the article's Figures 5–7.

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/matteogarbelli/ddpg-day-ahead-bidding.git
cd ddpg-day-ahead-bidding
pip install -e ".[dev]"
pytest
```

## Quick start

```bash
python scripts/train.py --config configs/episode5.json --seed 0   # 2.5 min (episode10: 5 min) on one CPU core
python scripts/evaluate.py runs/episode5_seed0                     # test-set reward, with a baseline
python scripts/plot.py runs/episode5_seed0                         # figures in runs/episode5_seed0/figures
bash scripts/run_all.sh                                            # every config, seeds 0-2
```

`train.py` writes to `runs/<config>_seed<seed>/` the resolved configuration, the per-episode
metrics (`metrics.csv`), the test-set evaluations during training (`test_evaluations.csv`) and
the network weights.

## Model

**Time.** A step is one day $d$. After observing the prices of day $d$, the agent submits a
curve for hour $h$ of day $d+1$ (default $h$ = 13:00–14:00). It is paid according to
$\mathrm{PUN}_{d+1,h}$. An episode is a sequence of $L$ consecutive days.

**State.** $s_d = \big(\mathrm{PUN}_{d-6},\dots,\mathrm{PUN}_{d};\ C_1,\dots,C_K;\ D_1,\dots,D_K\big)$:
the $7\times 24$ hourly prices of the last seven days, the unit production costs $C_k$ and the
capacities $D_k$. The supplier is a price taker, so $s_{d+1}$ does not depend on the action.

**Action.** An offering curve with $I$ steps $(P_i, V_i)$, $P_1 \le \dots \le P_I$ and
$0 \le V_i \le b_i$. With one step per production mode ($I = K$), $b_i$ is the capacity of the
$i$-th cheapest mode; otherwise $b_i = \sum_k D_k / I$.

**Reward** (Eq. (reward) of the article). The accepted volume is
$Q = \sum_i V_i\, \mathbb{1}\{P_i \le \mathrm{PUN}_{d+1,h}\}$. It is produced in merit order,
$q_k(Q)$ from the cheapest mode up. The profit is

$$
r = \sum_{i=1}^{I} P_i V_i\, \mathbb{1}\{P_i \le \mathrm{PUN}_{d+1,h}\} - \sum_{k=1}^{K} C_k\, q_k(Q).
$$

With `pricing: "clearing"` the accepted volume is paid at the PUN: $\mathrm{PUN}_{d+1,h}\,Q$.

**Normalised reward.** The largest profit attainable at the realised price sells every
profitable mode in full at the PUN:

$$
r_{\max} = \sum_{k=1}^{K} D_k\,(\mathrm{PUN}_{d+1,h} - C_k)^+ , \qquad r \le r_{\max}.
$$

The agent is trained on $r / \max(r_{\max}, \underline r)$, where
$\underline r = 1\ \text{EUR/MWh} \cdot \sum_k D_k$ keeps the ratio defined on the days when
no mode is profitable ($\mathrm{PUN}_{d+1,h} \le \min_k C_k$).

**Train/test split.** Target days are grouped in blocks of 14 consecutive days. In each
(year, quarter), 20% of the blocks (at least one) are held out for testing. Training episodes
use training target days only.

**Agent.** DDPG (Lillicrap et al., 2015; Algorithm 1 of the article): actor and critic are
feed-forward networks with two hidden layers of 64 ReLU units, with target networks updated by
Polyak averaging. The critic returns a scalar $Q(s,a)$. Exploration adds an Ornstein–Uhlenbeck
process, discretised by Euler–Maruyama (Eq. (discr_ou)), to the actor's curve, which is then
projected back to the feasible set. The actor's prices are
$P_i = \mathrm{PUN}_{d,h} + 50\tanh(z_i)$ EUR/MWh, clipped to $[0, 200]$ and sorted, where
$\mathrm{PUN}_{d,h}$ is the last observed price of the target hour; the volumes are
$V_i = b_i\,\sigma(y_i)$.

**Baseline.** `evaluate.py` also reports a persistence strategy: each mode's full capacity is
offered at $\max(C_k, \mathrm{PUN}_{d,h})$.

## Settings

The three configurations differ only in the episode length. `configs/article_table1.json` holds
the values of Table 1 of the article, with prices parameterised on $[0, 200]$ EUR/MWh.

| Parameter | Article, Table 1 | `configs/episode*.json` |
|---|---|---|
| Episodes | 1000, 1500 | 1000 |
| Episode length (days) | 5, 6, 10 | 5, 7, 10 |
| Batch size | 64 | 64 |
| Hidden units | 64 | 64 (two layers) |
| Actor learning rate | 1e-4 | 1e-4 |
| Critic learning rate | 1e-5 | 1e-3 |
| Discount factor $\gamma$ | 0.99 | 0 |
| Soft update $\tau$ | 0.01 | 0.01 |
| Replay memory | 50 000 | 50 000 |
| Gradient steps per environment step | 1 | 20 |
| L2 regularisation | yes | 1e-4 (actor and critic) |
| OU noise $\theta$, $\mu$, $\sigma$ | 0.15, 1, $\sigma \in [1, 10]$ | 0.15, 1, $\sigma$ from 10 to 1 linearly |
| Actor prices | — | anchored at $\mathrm{PUN}_{d,h}$, $\pm 50$ EUR/MWh |

Because the next state does not depend on the action, the action that maximises the expected
one-step reward is optimal for every $\gamma$; with $\gamma = 0$ the critic's target is the
reward alone.

## Results

<!-- RESULTS -->

## Repository layout

```
configs/            run manifests (JSON)
data/               hourly PUN 2017-2020 and its description
scripts/            prepare_data.py, train.py, evaluate.py, plot.py, run_all.sh
src/ddpg_bidding/
  config.py         configuration dataclasses
  data.py           price loading, stratified split
  market.py         clearing, profit, r_max, feasible curves
  env.py            day-ahead bidding environment
  networks.py       actor, critic, state encoding
  noise.py          Ornstein-Uhlenbeck process
  replay.py         replay buffer
  agent.py          DDPG update
  evaluate.py       policy evaluation, persistence baseline
  train.py          training loop
tests/              unit tests and a reproducibility test
```

## Extending the model

- **Stochastic costs and capacities.** `DayAheadBiddingEnv` reads `costs` and `capacities` at
  every step, and both enter the state; a time-dependent version only has to supply them per day.
- **Longer curves.** `market.n_steps` sets $I$; `market.step_bounds` sets the volume bounds.
- **Other encoders.** The actor and critic bodies in `networks.py` take the encoded state; a
  recurrent or convolutional encoder of the $7 \times 24$ price history can replace the first layer.
- **Other hours and pricing rules.** `data.target_hour` and `market.pricing`.

## Citation

See `CITATION.cff`, or

```bibtex
@article{DiPersio2025bidding,
  author  = {Di Persio, Luca and Garbelli, Matteo and Giordano, Luca Maria},
  title   = {Reinforcement learning for bidding strategy optimization in day-ahead energy market},
  journal = {Energy Economics},
  volume  = {149},
  pages   = {108673},
  year    = {2025},
  doi     = {10.1016/j.eneco.2025.108673}
}
```

## License

Code: MIT (see `LICENSE`). Data: see `data/README.md`.
