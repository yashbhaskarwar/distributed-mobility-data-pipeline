from __future__ import annotations
from pathlib import Path
from src.utils.config import load_config
from src.utils.delta_utils import read_delta
from src.utils.spark_session import get_spark_session

def assert_table(spark, path: str, min_rows: int = 1) -> None:
    df = read_delta(spark, path)
    n = df.count()
    assert n >= min_rows, f"Table at {path} has too few rows: {n}"

def main() -> None:
    cfg = load_config()
    gold = Path(cfg["paths"]["gold"])

    spark = get_spark_session(
        app_name="GoldTest",
        master=cfg["spark"]["master"],
        shuffle_partitions=10,
        timezone=cfg["spark"]["timezone"],
    )

    assert_table(spark, str(gold / "dim_time"), 10)
    assert_table(spark, str(gold / "dim_locations"), 10)
    assert_table(spark, str(gold / "dim_users"), 10)
    assert_table(spark, str(gold / "dim_drivers"), 10)
    assert_table(spark, str(gold / "dim_vehicle"), 10)

    assert_table(spark, str(gold / "trips_fact"), 1000)
    assert_table(spark, str(gold / "payments_fact"), 1000)
    spark.stop()

if __name__ == "__main__":
    main()