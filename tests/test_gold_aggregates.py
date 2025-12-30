from __future__ import annotations
from pathlib import Path
from src.utils.config import load_config
from src.utils.delta_utils import read_delta
from src.utils.spark_session import get_spark_session

def assert_min_rows(spark, path: str, min_rows: int) -> None:
    df = read_delta(spark, path)
    n = df.count()
    assert n >= min_rows, f"Too few rows at {path}: {n}"

def main() -> None:
    cfg = load_config()
    agg_root = Path(cfg["paths"]["gold"]) / "aggregates"

    spark = get_spark_session(
        app_name="AggTest",
        master=cfg["spark"]["master"],
        shuffle_partitions=10,
        timezone=cfg["spark"]["timezone"],
    )

    assert_min_rows(spark, str(agg_root / "demand_hourly_by_pickup_zone"), 100)
    assert_min_rows(spark, str(agg_root / "revenue_daily_by_pickup_zone"), 50)
    assert_min_rows(spark, str(agg_root / "driver_daily_summary"), 50)

    print("Gold aggregate tables exist.")
    spark.stop()

if __name__ == "__main__":
    main()
    