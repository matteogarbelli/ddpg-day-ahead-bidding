# Data

`pun_hourly_2017_2020.csv` contains the hourly Italian single national price (PUN,
*Prezzo Unico Nazionale*) of the day-ahead market (MGP), in EUR/MWh, from 1 January 2017
to 31 December 2020: one row per day (1461 rows), columns `date, h01, ..., h24`, where
`hXX` is the delivery hour ending at XX:00 local time.

**Source:** GME – Gestore dei Mercati Energetici S.p.A., historical MGP prices,
<https://www.mercatoelettrico.org>. Use of the data is subject to GME's terms of use.

**Processing** (`scripts/prepare_data.py`), from the yearly GME exports (`<year>comma.csv`,
semicolon separated, decimal comma):

- only the `PUN` column is kept;
- on the 23-hour day of the spring clock change, the missing 02:00–03:00 value is the mean
  of the two adjacent hours;
- on the 25-hour day of the autumn clock change, the two 02:00–03:00 values are averaged.

To rebuild the file, or to extend it to other years:

```bash
python scripts/prepare_data.py --raw-dir <folder with 2017comma.csv, ...> --years 2017 2020 \
    --out data/pun_hourly_2017_2020.csv
```
