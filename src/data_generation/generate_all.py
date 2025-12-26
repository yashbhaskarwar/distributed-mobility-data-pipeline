from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _choose_weighted(items: List[str], weights: List[float], n: int) -> List[str]:
    return random.choices(items, weights=weights, k=n)


def _generate_timestamps(start_date: datetime, days: int, n: int) -> pd.Series:
    total_minutes = days * 24 * 60
    day_offsets = np.random.randint(0, days, size=n)

    # Sample hour with peak weights
    hours = np.array(
        _choose_weighted(
            items=[str(h) for h in range(24)],
            weights=[
                0.6, 0.4, 0.3, 0.3, 0.4, 0.7,  # 0-5
                1.4, 2.2, 2.0, 1.3, 1.0, 1.0,  # 6-11
                1.1, 1.0, 1.0, 1.2, 1.6, 2.1,  # 12-17
                2.3, 2.0, 1.6, 1.2, 0.9, 0.7   # 18-23
            ],
            n=n,
        )
    ).astype(int)

    minutes = np.random.randint(0, 60, size=n)
    seconds = np.random.randint(0, 60, size=n)

    ts = [
        start_date + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(day_offsets, hours, minutes, seconds)
    ]
    return pd.Series(pd.to_datetime(ts), name="requested_at")


def generate_trips(cfg: Dict) -> pd.DataFrame:
    gen = cfg["generator"]
    paths = cfg["paths"]

    n_trips = int(gen["num_trips"])
    num_users = int(gen["num_users"])
    num_drivers = int(gen["num_drivers"])
    days = int(gen["days"])

    logger.info(f"Generating trips: n_trips={n_trips}, users={num_users}, drivers={num_drivers}, days={days}")

    start_date = datetime.utcnow() - timedelta(days=days)

    trip_id = np.arange(1, n_trips + 1)
    user_id = np.random.randint(1, num_users + 1, size=n_trips)
    driver_id = np.random.randint(1, num_drivers + 1, size=n_trips)

    # Simulate pickup/dropoff zones
    num_zones = 200
    pickup_zone = np.random.randint(1, num_zones + 1, size=n_trips)
    dropoff_zone = np.random.randint(1, num_zones + 1, size=n_trips)

    requested_at = _generate_timestamps(start_date, days, n_trips)

    # Trip distance 
    distance_km = np.clip(np.random.lognormal(mean=1.2, sigma=0.6, size=n_trips), 0.5, 60.0)

    # Duration in minutes 
    base_speed_kmph = np.random.normal(loc=28.0, scale=6.0, size=n_trips)  # city traffic
    base_speed_kmph = np.clip(base_speed_kmph, 10.0, 55.0)
    duration_min = (distance_km / base_speed_kmph) * 60.0
    duration_min = duration_min * np.random.normal(loc=1.0, scale=0.15, size=n_trips)
    duration_min = np.clip(duration_min, 3.0, 180.0)

    started_at = requested_at + pd.to_timedelta(np.random.randint(1, 12, size=n_trips), unit="m")
    completed_at = started_at + pd.to_timedelta(duration_min, unit="m")

    # Trip status
    status = _choose_weighted(
        items=["completed", "cancelled_rider", "cancelled_driver"],
        weights=[0.92, 0.06, 0.02],
        n=n_trips,
    )

    # Surge 
    hours = requested_at.dt.hour.values
    peak = np.isin(hours, [7, 8, 9, 17, 18, 19, 20]).astype(float)
    surge = 1.0 + peak * np.random.beta(a=2.5, b=6.0, size=n_trips) * 1.5
    surge = np.clip(surge, 1.0, 3.5)

    # Fare = base + per-km + per-minute
    base_fare = 2.5
    per_km = 1.15
    per_min = 0.35

    fare = (base_fare + per_km * distance_km + per_min * duration_min) * surge
    fare = fare * np.random.normal(loc=1.0, scale=0.05, size=n_trips)
    fare = np.round(np.clip(fare, 3.0, 250.0), 2)

    # Payment method distribution
    payment_method = _choose_weighted(
        items=["card", "wallet", "cash"],
        weights=[0.78, 0.18, 0.04],
        n=n_trips,
    )

    df = pd.DataFrame(
        {
            "trip_id": trip_id,
            "user_id": user_id,
            "driver_id": driver_id,
            "pickup_zone_id": pickup_zone,
            "dropoff_zone_id": dropoff_zone,
            "requested_at": requested_at.astype("datetime64[ns]"),
            "started_at": started_at.astype("datetime64[ns]"),
            "completed_at": completed_at.astype("datetime64[ns]"),
            "status": status,
            "distance_km": np.round(distance_km, 2),
            "duration_min": np.round(duration_min, 2),
            "surge_multiplier": np.round(surge, 2),
            "fare_amount": fare,
            "payment_method": payment_method,
        }
    )

    return df


def write_raw_csv(df: pd.DataFrame, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Wrote raw CSV: {p} rows={len(df)}")


def main() -> None:
    cfg = load_config()
    raw_dir = cfg["paths"]["raw"]
    out_file = str(Path(raw_dir) / "trips.csv")

    df = generate_trips(cfg)
    write_raw_csv(df, out_file)

    # Summary
    print(df.head(3).to_string(index=False))
    print("Row count:", len(df))
    print("Date range:", df["requested_at"].min(), "to", df["requested_at"].max())
    print("Status distribution:")
    print(df["status"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
