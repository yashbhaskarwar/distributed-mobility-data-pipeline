from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, lower, trim, row_number, to_timestamp, when

from src.utils.config import load_config
from src.utils.delta_utils import read_delta, write_delta_overwrite
from src.utils.logger import setup_logger
from src.utils.spark_session import get_spark_session
from src.utils.data_quality import run_silver_checks

logger = setup_logger(__name__)


def clean_trips(df: DataFrame) -> DataFrame:
    # Normalize strings
    df = df.withColumn("status", lower(trim(col("status"))))
    df = df.withColumn("payment_method", lower(trim(col("payment_method"))))

    df = (
        df.withColumn("requested_at", col("requested_at").cast("timestamp"))
        .withColumn("started_at", col("started_at").cast("timestamp"))
        .withColumn("completed_at", col("completed_at").cast("timestamp"))
    )

    df = df.withColumn(
        "completed_at",
        when(col("status").startswith("cancelled"), None).otherwise(col("completed_at")),
    )

    df = df.filter(
        col("trip_id").isNotNull()
        & col("user_id").isNotNull()
        & col("driver_id").isNotNull()
        & col("requested_at").isNotNull()
    )

    w = Window.partitionBy("trip_id").orderBy(col("ingested_at").desc())
    df = df.withColumn("rn", row_number().over(w)).filter(col("rn") == 1).drop("rn")

    return df

def main() -> None:
    cfg = load_config()
    bronze_path = str(Path(cfg["paths"]["bronze"]) / "trips")
    silver_path = str(Path(cfg["paths"]["silver"]) / "trips")

    spark = get_spark_session(
        app_name=cfg["spark"]["app_name"],
        master=cfg["spark"]["master"],
        shuffle_partitions=int(cfg["spark"]["shuffle_partitions"]),
        timezone=cfg["spark"]["timezone"],
    )

    logger.info(f"Reading Bronze Delta: {bronze_path}")
    bronze_df = read_delta(spark, bronze_path)

    logger.info("Cleaning trips for Silver layer")
    silver_df = clean_trips(bronze_df)

    logger.info(f"Silver rows: {silver_df.count()}")

    # Data quality checks
    results = run_silver_checks(silver_df)
    if not all(r.passed for r in results):
        raise RuntimeError("Silver data quality checks failed")

    logger.info(f"Writing Silver Delta: {silver_path}")
    # Partition by requested_date retained from Bronze 
    write_delta_overwrite(silver_df, silver_path, partition_by=["requested_date"])

    logger.info("Bronze to Silver transformation complete")
    spark.stop()

if __name__ == "__main__":
    main()