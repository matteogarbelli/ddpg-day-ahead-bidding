# DDPG bidding in the day-ahead electricity market

Code accompanying

> L. Di Persio, M. Garbelli, L. M. Giordano, *Reinforcement learning for bidding strategy
> optimization in day-ahead energy market*, Energy Economics 149 (2025) 108673.
> [doi:10.1016/j.eneco.2025.108673](https://doi.org/10.1016/j.eneco.2025.108673)

A supplier with three production modes submits, every day, a stepwise offering curve for one
delivery hour of the next day in the Italian day-ahead market. A Deep Deterministic Policy
Gradient (DDPG) agent chooses the curve from the hourly national prices (PUN) of the last seven
days.

## Installation

```bash
git clone https://github.com/matteogarbelli/ddpg-day-ahead-bidding.git
cd ddpg-day-ahead-bidding
uv sync && source .venv/bin/activate   # exact versions from uv.lock
pytest
```

Without [uv](https://docs.astral.sh/uv/): `pip install -e .` (Python ≥ 3.10, versions not pinned).

## Reproducing the results

```bash
bash scripts/run_all.sh
```

This command trains every configuration in `configs/` with seeds 0, 1 and 2, running the nine
runs in parallel. It then prints the normalised reward on the held-out days: the profit divided
by the largest profit attainable at the realised price (`normalized_reward` in
`src/ddpg_bidding/market.py`).

| Configuration | Episode length (days) | DDPG | Persistence baseline |
|---|---|---|---|
| `episode5` | 5 | 0.497 ± 0.008 | 0.303 |
| `episode7` | 7 | 0.496 ± 0.005 | 0.303 |
| `episode10` | 10 | 0.435 ± 0.113 | 0.303 |

DDPG: mean ± standard deviation over the three seeds. The persistence baseline offers each
mode's full capacity at the larger of its cost and the previous day's price.

A single run:

```bash
python scripts/train.py --config configs/episode5.json --seed 0   # writes runs/episode5_seed0/
python scripts/evaluate.py runs/episode5_seed0
python scripts/plot.py runs/episode5_seed0                         # runs/episode5_seed0/figures/
```

## Data

`data/pun_hourly_2017_2020.csv` holds the hourly PUN from 2017 to 2020, published by GME
(<https://www.mercatoelettrico.org>). See `data/README.md` for the source, the terms of use
and the processing.

## Citation

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

The code is released under the MIT license (see `LICENSE`). The data are subject to GME's terms of use.
