from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql.functions import current_timestamp, input_file_name, to_date, col

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.spark_session import get_spark_session
from src.utils.delta_utils import write_delta_overwrite

logger = setup_logger(__name__)

def read_raw_trips(spark, raw_path: str) -> DataFrame:
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(raw_path)
    )

    df = (
        df.withColumn("requested_at", col("requested_at").cast("timestamp"))
        .withColumn("started_at", col("started_at").cast("timestamp"))
        .withColumn("completed_at", col("completed_at").cast("timestamp"))
    )
    return df


def add_audit_columns(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("ingested_at", current_timestamp())
        .withColumn("source_file", input_file_name())
        .withColumn("requested_date", to_date(col("requested_at")))
    )

def main() -> None:
    cfg = load_config()
    raw_file = str(Path(cfg["paths"]["raw"]) / "trips.csv")
    bronze_path = str(Path(cfg["paths"]["bronze"]) / "trips")

    spark = get_spark_session(
        app_name=cfg["spark"]["app_name"],
        master=cfg["spark"]["master"],
        shuffle_partitions=int(cfg["spark"]["shuffle_partitions"]),
        timezone=cfg["spark"]["timezone"],
    )

    logger.info(f"Reading raw trips: {raw_file}")
    df = read_raw_trips(spark, raw_file)
    df = add_audit_columns(df)

    logger.info(f"Rows read: {df.count()}")
    logger.info(f"Writing Bronze Delta: {bronze_path}")

    write_delta_overwrite(df, bronze_path, partition_by=["requested_date"])

    logger.info("Bronze ingestion complete")
    spark.stop()

if __name__ == "__main__":
    main()