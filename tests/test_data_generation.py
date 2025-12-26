from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    raw_path = Path(cfg["paths"]["raw"]) / "trips.csv"

    assert raw_path.exists(), f"Missing file: {raw_path}. Run generator first."

    df = pd.read_csv(raw_path)
    required_cols = {
        "trip_id",
        "user_id",
        "driver_id",
        "pickup_zone_id",
        "dropoff_zone_id",
        "requested_at",
        "started_at",
        "completed_at",
        "status",
        "distance_km",
        "duration_min",
        "surge_multiplier",
        "fare_amount",
        "payment_method",
    }

    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"
    assert df["trip_id"].is_unique, "trip_id should be unique"
    assert len(df) > 1000, "Too few rows generated"
    assert df["fare_amount"].min() >= 0, "Fare should be non-negative"

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

if __name__ == "__main__":
    main()
    