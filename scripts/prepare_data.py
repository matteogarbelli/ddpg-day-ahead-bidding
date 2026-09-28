"""Build the hourly PUN table from GME yearly price exports.

Input: one semicolon-separated file per year, as exported from the GME historical
data page (https://www.mercatoelettrico.org), with columns
``Data/Date (YYYYMMDD)``, ``Ora/Hour`` (1-23, 1-24 or 1-25) and ``PUN`` (decimal comma).

Output: a wide CSV with one row per day and columns ``date, h01, ..., h24``.
Clock-change days are mapped to 24 values:

* 23-hour day (last Sunday of March): the missing 02:00-03:00 value is the mean of
  the adjacent hours;
* 25-hour day (last Sunday of October): the two 02:00-03:00 values are averaged.

Usage::

    python scripts/prepare_data.py --raw-dir <folder with 2017comma.csv ...> \
        --years 2017 2020 --out data/pun_hourly_2017_2020.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_gme_year(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "hour"})
    df = df[["date", "hour", "PUN"]]
    df["PUN"] = df["PUN"].astype(str).str.replace(",", ".").astype(float)
    return df


def day_to_24(values: np.ndarray) -> np.ndarray:
    """Map the hourly values of one delivery day to 24 clock hours."""
    n = len(values)
    if n == 24:
        return values
    if n == 23:
        return np.concatenate([values[:2], [0.5 * (values[1] + values[2])], values[2:]])
    if n == 25:
        return np.concatenate([values[:2], [0.5 * (values[2] + values[3])], values[4:]])
    raise ValueError(f"unexpected number of hours in a day: {n}")


def build_table(raw_dir: Path, first_year: int, last_year: int) -> pd.DataFrame:
    rows = []
    for year in range(first_year, last_year + 1):
        df = read_gme_year(raw_dir / f"{year}comma.csv")
        for date, group in df.groupby("date", sort=True):
            values = group.sort_values("hour")["PUN"].to_numpy()
            rows.append([pd.to_datetime(str(date), format="%Y%m%d").date()] + list(day_to_24(values)))
    columns = ["date"] + [f"h{h:02d}" for h in range(1, 25)]
    table = pd.DataFrame(rows, columns=columns)
    expected = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D").date
    if list(table["date"]) != list(expected):
        raise ValueError("the exported files do not cover every calendar day")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs=2, default=(2017, 2020), metavar=("FIRST", "LAST"))
    parser.add_argument("--out", type=Path, default=Path("data/pun_hourly_2017_2020.csv"))
    args = parser.parse_args()

    table = build_table(args.raw_dir, *args.years)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.2f")
    print(f"wrote {len(table)} days to {args.out}")


if __name__ == "__main__":
    main()
