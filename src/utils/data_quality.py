from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnan, isnull, count, sum as spark_sum
from src.utils.logger import setup_logger
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnan, sum as Fsum, when
from pyspark.sql.types import DoubleType, FloatType, DecimalType, IntegerType, LongType, ShortType

logger = setup_logger(__name__)

@dataclass
class QualityResult:
    name: str
    passed: bool
    details: str

_NUMERIC_TYPES = (DoubleType, FloatType, DecimalType, IntegerType, LongType, ShortType)

def _null_count(df: DataFrame, col_name: str) -> int:
    field = next((f for f in df.schema.fields if f.name == col_name), None)
    if field is None:
        raise ValueError(f"Column not found: {col_name}")

    c = col(col_name)

    if isinstance(field.dataType, _NUMERIC_TYPES):
        expr = c.isNull() | isnan(c)
    else:
        expr = c.isNull()

    row = df.select(Fsum(when(expr, 1).otherwise(0)).alias("nulls")).collect()[0]
    return int(row["nulls"])

def check_required_columns(df: DataFrame, required: List[str]) -> QualityResult:
    missing = [c for c in required if c not in df.columns]
    passed = len(missing) == 0
    details = "OK" if passed else f"Missing columns: {missing}"
    return QualityResult("required_columns", passed, details)


def check_non_null(df: DataFrame, columns: List[str], max_null_rate: float = 0.001) -> QualityResult:
    total = df.count()
    if total == 0:
        return QualityResult("non_null", False, "Empty dataset")

    bad = []
    for c in columns:
        n_null = _null_count(df, c)
        rate = n_null / total
        if rate > max_null_rate:
            bad.append((c, n_null, rate))

    passed = len(bad) == 0
    details = "OK" if passed else f"High null rate: {bad}"
    return QualityResult("non_null", passed, details)

def check_positive(df: DataFrame, columns: List[str]) -> QualityResult:
    bad = []
    for c in columns:
        n = df.filter(col(c) < 0).count()
        if n > 0:
            bad.append((c, n))
    passed = len(bad) == 0
    details = "OK" if passed else f"Negative values found: {bad}"
    return QualityResult("positive_values", passed, details)

def run_silver_checks(df: DataFrame) -> List[QualityResult]:
    required = [
        "trip_id",
        "user_id",
        "driver_id",
        "pickup_zone_id",
        "dropoff_zone_id",
        "requested_at",
        "status",
        "fare_amount",
        "distance_km",
        "duration_min",
        "surge_multiplier",
    ]

    results = []
    results.append(check_required_columns(df, required))
    results.append(check_non_null(df, ["trip_id", "user_id", "driver_id", "requested_at", "status"]))
    results.append(check_positive(df, ["fare_amount", "distance_km", "duration_min", "surge_multiplier"]))

    for r in results:
        if r.passed:
            logger.info(f"DQ PASS: {r.name} | {r.details}")
        else:
            logger.error(f"DQ FAIL: {r.name} | {r.details}")

    return results
