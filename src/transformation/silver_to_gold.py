from __future__ import annotations
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    date_trunc,
    to_date,
    year,
    month,
    dayofmonth,
    dayofweek,
    hour,
    minute,
    lit,
    sha2,
    concat_ws,
)

from src.utils.config import load_config
from src.utils.delta_utils import read_delta, write_delta_overwrite
from src.utils.logger import setup_logger
from src.utils.spark_session import get_spark_session

logger = setup_logger(__name__)


def build_dim_time(silver: DataFrame) -> DataFrame:
    df = silver.select(col("requested_at")).where(col("requested_at").isNotNull()).dropDuplicates()
    df = (
        df.withColumn("date", to_date(col("requested_at")))
        .withColumn("year", year(col("requested_at")))
        .withColumn("month", month(col("requested_at")))
        .withColumn("day", dayofmonth(col("requested_at")))
        .withColumn("day_of_week", dayofweek(col("requested_at")))  
        .withColumn("hour", hour(col("requested_at")))
        .withColumn("minute", minute(col("requested_at")))
        .select("date", "year", "month", "day", "day_of_week", "hour", "minute")
        .dropDuplicates()
    )
    df = df.withColumn(
        "time_id",
        sha2(concat_ws("||", col("date").cast("string"), col("hour").cast("string"), col("minute").cast("string")), 256),
    )
    return df.select("time_id", "date", "year", "month", "day", "day_of_week", "hour", "minute")


def build_dim_locations(silver: DataFrame) -> DataFrame:
    # Using zone ids as locations
    df = (
        silver.select(
            col("pickup_zone_id").alias("zone_id")
        )
        .union(silver.select(col("dropoff_zone_id").alias("zone_id")))
        .dropna()
        .dropDuplicates()
    )

    df = df.withColumn("city", lit("synthetic_city")).withColumn("zone_type", lit("zone"))
    df = df.withColumn("location_id", sha2(col("zone_id").cast("string"), 256))
    return df.select("location_id", "zone_id", "city", "zone_type")


def build_dim_users(silver: DataFrame) -> DataFrame:
    df = silver.select("user_id").where(col("user_id").isNotNull()).dropDuplicates()
    df = df.withColumn("user_key", sha2(col("user_id").cast("string"), 256))
    return df.select(col("user_key").alias("user_key"), "user_id")


def build_dim_drivers(silver: DataFrame) -> DataFrame:
    df = silver.select("driver_id").where(col("driver_id").isNotNull()).dropDuplicates()
    df = df.withColumn("driver_key", sha2(col("driver_id").cast("string"), 256))
    return df.select(col("driver_key").alias("driver_key"), "driver_id")


def build_dim_vehicle(silver: DataFrame) -> DataFrame:
    df = silver.select("driver_id").where(col("driver_id").isNotNull()).dropDuplicates()
    df = (
        df.withColumn("vehicle_type", lit("standard"))
        .withColumn("vehicle_key", sha2(col("driver_id").cast("string"), 256))
    )
    return df.select("vehicle_key", "driver_id", "vehicle_type")


def build_trips_fact(
    silver: DataFrame,
    dim_time: DataFrame,
    dim_locations: DataFrame,
    dim_users: DataFrame,
    dim_drivers: DataFrame,
    dim_vehicle: DataFrame,
) -> DataFrame:
    # Join keys
    s = silver
    # time_id join
    s_time = s.withColumn(
        "time_id",
        sha2(concat_ws("||", to_date(col("requested_at")).cast("string"), hour(col("requested_at")).cast("string"), minute(col("requested_at")).cast("string")), 256),
    )

    # location keys
    pickup_loc = dim_locations.select(
        col("location_id").alias("pickup_location_id"),
        col("zone_id").alias("pickup_zone_id"),
    )
    dropoff_loc = dim_locations.select(
        col("location_id").alias("dropoff_location_id"),
        col("zone_id").alias("dropoff_zone_id"),
    )

    df = (
        s_time.join(dim_users.withColumnRenamed("user_key", "user_key"), on="user_id", how="left")
        .join(dim_drivers.withColumnRenamed("driver_key", "driver_key"), on="driver_id", how="left")
        .join(dim_vehicle.select("vehicle_key", "driver_id"), on="driver_id", how="left")
        .join(pickup_loc, on="pickup_zone_id", how="left")
        .join(dropoff_loc, on="dropoff_zone_id", how="left")
    )

    fact = df.select(
        col("trip_id"),
        col("time_id"),
        col("user_key"),
        col("driver_key"),
        col("vehicle_key"),
        col("pickup_location_id"),
        col("dropoff_location_id"),
        col("status"),
        col("distance_km"),
        col("duration_min"),
        col("surge_multiplier"),
        col("fare_amount"),
        col("requested_at"),
        col("started_at"),
        col("completed_at"),
        col("requested_date"),
    )

    return fact


def build_payments_fact(silver: DataFrame, dim_time: DataFrame, dim_users: DataFrame, dim_drivers: DataFrame) -> DataFrame:
    s = silver

    s = s.withColumn(
        "time_id",
        sha2(concat_ws("||", to_date(col("requested_at")).cast("string"), hour(col("requested_at")).cast("string"), minute(col("requested_at")).cast("string")), 256),
    )
    df = (
        s.join(dim_users, on="user_id", how="left")
        .join(dim_drivers, on="driver_id", how="left")
    )

    fact = df.select(
        col("trip_id").alias("payment_id"),
        col("trip_id"),
        col("time_id"),
        col("user_key"),
        col("driver_key"),
        col("payment_method"),
        col("fare_amount").alias("amount"),
        col("requested_date"),
    )
    return fact

def main() -> None:
    cfg = load_config()
    silver_path = str(Path(cfg["paths"]["silver"]) / "trips")
    gold_root = Path(cfg["paths"]["gold"])

    spark = get_spark_session(
        app_name=cfg["spark"]["app_name"],
        master=cfg["spark"]["master"],
        shuffle_partitions=int(cfg["spark"]["shuffle_partitions"]),
        timezone=cfg["spark"]["timezone"],
    )

    logger.info(f"Reading Silver Delta: {silver_path}")
    silver = read_delta(spark, silver_path)

    logger.info("Building dimensions")
    dim_time = build_dim_time(silver)
    dim_locations = build_dim_locations(silver)
    dim_users = build_dim_users(silver)
    dim_drivers = build_dim_drivers(silver)
    dim_vehicle = build_dim_vehicle(silver)

    logger.info("Building facts")
    trips_fact = build_trips_fact(silver, dim_time, dim_locations, dim_users, dim_drivers, dim_vehicle)
    payments_fact = build_payments_fact(silver, dim_time, dim_users, dim_drivers)

    # Gold tables
    write_delta_overwrite(dim_time, str(gold_root / "dim_time"))
    write_delta_overwrite(dim_locations, str(gold_root / "dim_locations"))
    write_delta_overwrite(dim_users, str(gold_root / "dim_users"))
    write_delta_overwrite(dim_drivers, str(gold_root / "dim_drivers"))
    write_delta_overwrite(dim_vehicle, str(gold_root / "dim_vehicle"))

    write_delta_overwrite(trips_fact, str(gold_root / "trips_fact"), partition_by=["requested_date"])
    write_delta_overwrite(payments_fact, str(gold_root / "payments_fact"), partition_by=["requested_date"])

    logger.info("Silver to Gold transformation complete")
    spark.stop()

if __name__ == "__main__":
    main()