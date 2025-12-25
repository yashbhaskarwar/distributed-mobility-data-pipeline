from __future__ import annotations

from pyspark.sql import SparkSession

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_spark_session(
    app_name: str = "MobilityPipeline",
    master: str = "local[*]",
    shuffle_partitions: int = 200,
    timezone: str = "UTC",
) -> SparkSession:
    logger.info("Creating Spark session")
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", timezone)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session ready")
    return spark