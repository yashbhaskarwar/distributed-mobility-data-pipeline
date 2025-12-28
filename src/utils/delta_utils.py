from __future__ import annotations

from pathlib import Path
from typing import Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def is_delta_table(spark: SparkSession, path: str) -> bool:
    try:
        return DeltaTable.isDeltaTable(spark, path)
    except Exception:
        return False


def read_delta(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.format("delta").load(path)

def write_delta_overwrite(
    df: DataFrame,
    path: str,
    partition_by: Optional[list[str]] = None,
) -> None:
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)
    logger.info(f"Wrote Delta table (overwrite): {path}")


def ensure_path(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    