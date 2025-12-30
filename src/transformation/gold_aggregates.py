from __future__ import annotations
from pathlib import Path
from pyspark.sql.functions import col, count, sum as spark_sum, avg
from src.utils.config import load_config
from src.utils.delta_utils import read_delta, write_delta_overwrite
from src.utils.logger import setup_logger
from src.utils.spark_session import get_spark_session

logger = setup_logger(__name__)

def main() -> None:
    cfg = load_config()
    gold_root = Path(cfg["paths"]["gold"])
    agg_root = gold_root / "aggregates"

    spark = get_spark_session(
        app_name="MobilityGoldAggregates",
        master=cfg["spark"]["master"],
        shuffle_partitions=int(cfg["spark"]["shuffle_partitions"]),
        timezone=cfg["spark"]["timezone"],
    )

    trips = read_delta(spark, str(gold_root / "trips_fact"))
    dim_time = read_delta(spark, str(gold_root / "dim_time"))
    dim_locations = read_delta(spark, str(gold_root / "dim_locations"))

    # Enrich trips 
    trips_enriched = (
        trips.join(dim_time.select("time_id", "date", "hour"), on="time_id", how="left")
        .join(
            dim_locations.select(
                col("location_id").alias("pickup_location_id"),
                col("zone_id").alias("pickup_zone_id"),
            ),
            on="pickup_location_id",
            how="left",
        )
    )

    # Demand: hourly trip volume by pickup zone
    demand_hourly_by_pickup_zone = (
        trips_enriched.groupBy("date", "hour", "pickup_zone_id")
        .agg(count("*").alias("trip_count"))
    )

    # Revenue: daily revenue and surge stats by pickup zone 
    completed = trips_enriched.filter(col("status") == "completed")
    revenue_daily_by_pickup_zone = (
        completed.groupBy("date", "pickup_zone_id")
        .agg(
            spark_sum(col("fare_amount")).alias("total_revenue"),
            avg(col("surge_multiplier")).alias("avg_surge_multiplier"),
            avg(col("distance_km")).alias("avg_distance_km"),
            count("*").alias("completed_trips"),
        )
    )

    # Driver daily summary
    driver_daily_summary = (
        completed.groupBy("requested_date", "driver_key")
        .agg(
            count("*").alias("completed_trips"),
            spark_sum(col("fare_amount")).alias("driver_revenue"),
            avg(col("duration_min")).alias("avg_trip_duration_min"),
        )
        .withColumnRenamed("requested_date", "date")
    )

    # Write aggregates
    write_delta_overwrite(
        demand_hourly_by_pickup_zone,
        str(agg_root / "demand_hourly_by_pickup_zone"),
        partition_by=["date"],
    )
    write_delta_overwrite(
        revenue_daily_by_pickup_zone,
        str(agg_root / "revenue_daily_by_pickup_zone"),
        partition_by=["date"],
    )
    write_delta_overwrite(
        driver_daily_summary,
        str(agg_root / "driver_daily_summary"),
        partition_by=["date"],
    )

    logger.info("Gold aggregates built successfully")
    spark.stop()

if __name__ == "__main__":
    main()