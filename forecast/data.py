"""Synthetic data generator.

Creates four tables that mirror what a real multi-outlet business has:

* locations.csv     - one row per outlet (area, store type, capacity)
* holidays.csv      - calendar of public / festival holidays
* reservations.csv  - advance bookings per outlet per day (known before the day)
* sales.csv         - actual daily demand (visitors) per outlet  <- the target

Reservations are also generated for the forecast horizon after the last sales
date, because in practice bookings for next week are already known today.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AREAS = ["Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune"]
STORE_TYPES = ["Cafe", "Casual Dining", "Quick Service", "Fine Dining"]

# Fixed-date national holidays + approximate festival dates (2023-2026)
HOLIDAYS = {
    "Republic Day": ["01-26"],
    "Independence Day": ["08-15"],
    "Gandhi Jayanti": ["10-02"],
    "Christmas": ["12-25"],
    "New Year": ["01-01"],
}
FESTIVALS = {
    "Holi": ["2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04"],
    "Diwali": ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"],
    "Eid al-Fitr": ["2023-04-22", "2024-04-11", "2025-03-31", "2026-03-20"],
    "Dussehra": ["2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20"],
    "Ganesh Chaturthi": ["2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14"],
}


def build_holidays(start: str, end: str) -> pd.DataFrame:
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    rows = []
    for name, mmdd in HOLIDAYS.items():
        for y in years:
            for d in mmdd:
                rows.append((pd.Timestamp(f"{y}-{d}"), name))
    for name, dates in FESTIVALS.items():
        rows.extend((pd.Timestamp(d), name) for d in dates)
    df = pd.DataFrame(rows, columns=["date", "holiday_name"])
    df = df[(df.date >= start) & (df.date <= end)].drop_duplicates("date")
    return df.sort_values("date").reset_index(drop=True)


def generate(
    n_locations: int = 12,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    horizon: int = 7,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    locations = pd.DataFrame(
        {
            "location_id": [f"L{i:03d}" for i in range(1, n_locations + 1)],
            "area": rng.choice(AREAS, n_locations),
            "store_type": rng.choice(STORE_TYPES, n_locations),
            "capacity": rng.integers(40, 200, n_locations),
        }
    )

    future_end = pd.Timestamp(end) + pd.Timedelta(days=horizon)
    holidays = build_holidays(start, str(future_end.date()))
    hol_set = set(holidays.date)

    dates = pd.date_range(start, future_end, freq="D")
    n_days = len(dates)
    t = np.arange(n_days)

    weekly = {0: 0.85, 1: 0.80, 2: 0.85, 3: 0.92, 4: 1.15, 5: 1.35, 6: 1.25}
    type_mult = {"Cafe": 0.8, "Casual Dining": 1.0, "Quick Service": 1.3, "Fine Dining": 0.6}

    sales_rows, res_rows = [], []
    for _, loc in locations.iterrows():
        base = loc.capacity * 0.45 * type_mult[loc.store_type]
        trend = 1 + rng.uniform(-0.05, 0.25) * t / n_days
        yearly = 1 + 0.12 * np.sin(2 * np.pi * (t - 60) / 365.25)
        dow = np.array([weekly[d.dayofweek] for d in dates])
        is_hol = np.array([d in hol_set for d in dates])
        pre_hol = np.array([(d + pd.Timedelta(days=1)) in hol_set for d in dates])
        hol_effect = np.where(is_hol, 1.35, 1.0) * np.where(pre_hol, 1.12, 1.0)
        # Fine dining gets busier on holidays, quick service less so
        if loc.store_type == "Fine Dining":
            hol_effect = hol_effect ** 1.5
        expected = base * trend * yearly * dow * hol_effect
        # AR(1) noise -> realistic momentum between days
        noise = np.zeros(n_days)
        for i in range(1, n_days):
            noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 0.08)
        demand = np.maximum(0, expected * np.exp(noise)).round().astype(int)
        # Bookings: share of demand that books ahead depends on store type
        book_rate = {"Fine Dining": 0.55, "Casual Dining": 0.3, "Cafe": 0.1, "Quick Service": 0.05}[
            loc.store_type
        ]
        reserves = rng.binomial(demand, book_rate)

        # ~1% of days closed (no sales row) to mimic real gaps
        closed = rng.random(n_days) < 0.01
        for i, d in enumerate(dates):
            if d <= pd.Timestamp(end):
                if not closed[i]:
                    sales_rows.append((d, loc.location_id, demand[i]))
            if reserves[i] > 0:
                res_rows.append((d, loc.location_id, int(reserves[i])))

    sales = pd.DataFrame(sales_rows, columns=["date", "location_id", "visitors"])
    reservations = pd.DataFrame(res_rows, columns=["date", "location_id", "reserve_visitors"])
    return {
        "locations": locations,
        "holidays": holidays,
        "reservations": reservations,
        "sales": sales,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic demand data")
    p.add_argument("--out", default="data")
    p.add_argument("--locations", type=int, default=12)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = generate(args.locations, args.start, args.end, seed=args.seed)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / f'{name}.csv'} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
