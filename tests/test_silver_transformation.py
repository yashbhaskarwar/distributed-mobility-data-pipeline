from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config
from src.utils.delta_utils import read_delta
from src.utils.spark_session import get_spark_session


def main() -> None:
    cfg = load_config()
    silver_path = str(Path(cfg["paths"]["silver"]) / "trips")

    spark = get_spark_session(
        app_name="SilverTest",
        master=cfg["spark"]["master"],
        shuffle_partitions=10,
        timezone=cfg["spark"]["timezone"],
    )

    df = read_delta(spark, silver_path)
    n = df.count()
    assert n > 1000, "Silver table has too few rows"

    # Basic invariants
    assert df.filter(df.trip_id.isNull()).count() == 0, "Null trip_id in Silver"
    assert df.filter(df.requested_at.isNull()).count() == 0, "Null requested_at in Silver"
    assert df.filter(df.fare_amount < 0).count() == 0, "Negative fare in Silver"
    print("Rows:", n)

    spark.stop()

if __name__ == "__main__":
    main()