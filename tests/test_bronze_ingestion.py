from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config
from src.utils.spark_session import get_spark_session
from src.utils.delta_utils import read_delta


def main() -> None:
    cfg = load_config()
    bronze_path = str(Path(cfg["paths"]["bronze"]) / "trips")

    spark = get_spark_session(
        app_name="BronzeTest",
        master=cfg["spark"]["master"],
        shuffle_partitions=10,
        timezone=cfg["spark"]["timezone"],
    )

    df = read_delta(spark, bronze_path)

    required_cols = {
        "trip_id",
        "user_id",
        "driver_id",
        "requested_at",
        "status",
        "fare_amount",
        "ingested_at",
        "source_file",
        "requested_date",
    }

    cols = set(df.columns)
    missing = required_cols - cols
    assert not missing, f"Missing columns in Bronze: {missing}"

    n = df.count()
    assert n > 1000, "Bronze table has too few rows"

    print("OK: Bronze Delta trips table exists")
    print("Rows:", n)
    print("Sample partition dates:", [r["requested_date"] for r in df.select("requested_date").distinct().limit(5).collect()])
    spark.stop()

if __name__ == "__main__":
    main()