import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

# Delta merge
from delta.tables import DeltaTable

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def build_spark(app_name: str = "mobility-batch-scoring") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

def default_paths() -> Dict[str, str]:
    root = repo_root()
    return {
        "silver_features": str(root / "data" / "features" / "silver_features"),
        "predictions_out": str(root / "data" / "gold" / "batch_predictions"),
        "metrics_out": str(root / "data" / "gold" / "batch_scoring_metrics"),
        "demand_model": str(root / "data" / "models" / "demand_forecasting" / "best_model.pkl"),
        "surge_model": str(root / "data" / "models" / "surge_pricing" / "best_model.pkl"),
    }

def load_local_model(model_path: str) -> Optional[Any]:
    p = Path(model_path)
    if not p.exists():
        return None
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        return None

def read_silver_features(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.format("delta").load(path)

def ensure_partition_cols(df: DataFrame) -> DataFrame:
    if "event_date" not in df.columns:
        # try to derive from a timestamp field if present
        if "event_ts" in df.columns:
            df = df.withColumn("event_date", F.to_date(F.col("event_ts")))
        elif "timestamp" in df.columns:
            df = df.withColumn("event_date", F.to_date(F.col("timestamp")))
        else:
            df = df.withColumn("event_date", F.to_date(F.lit(datetime.utcnow().strftime("%Y-%m-%d"))))

    if "city" not in df.columns:
        df = df.withColumn("city", F.lit("unknown"))

    return df

def score_demand(df: DataFrame, demand_model: Optional[Any]) -> DataFrame:
    required_cols = ["hour", "is_weekend", "demand_avg_7d", "avg_surge"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column for demand scoring: {c}")

    if demand_model is None:
        return df.withColumn(
            "predicted_demand",
            F.round(
                (F.col("demand_avg_7d")
                 * F.when(F.col("hour").isin([7, 8, 9, 16, 17, 18]), F.lit(1.25)).otherwise(F.lit(1.0))
                 * F.when(F.col("is_weekend") == 1, F.lit(0.85)).otherwise(F.lit(1.0))
                 * (F.lit(1.0) + F.greatest(F.col("avg_surge") - F.lit(1.0), F.lit(0.0)) * F.lit(0.15))
                 ),
                2,
            ),
        )

    feature_cols = [
        "hour",
        "is_weekend",
        "demand_lag_1",
        "demand_avg_7d",
        "avg_fare",
        "avg_surge",
    ]

    for c in feature_cols:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("double"))

    @pandas_udf(DoubleType())
    def predict_demand_udf(*cols: pd.Series) -> pd.Series:
        X = pd.concat(cols, axis=1)
        X.columns = feature_cols
        preds = demand_model.predict(X)
        return pd.Series(preds).astype(float)

    return df.withColumn("predicted_demand", F.round(predict_demand_udf(*[F.col(c) for c in feature_cols]), 2))


def score_surge(df: DataFrame, surge_model: Optional[Any]) -> DataFrame:
    if "predicted_demand" not in df.columns:
        raise ValueError("predicted_demand must exist before surge scoring")

    if "supply_index" not in df.columns:
        df = df.withColumn("supply_index", F.lit(1.0))
    if "rain_intensity" not in df.columns:
        df = df.withColumn("rain_intensity", F.lit(0.0))
    if "hour" not in df.columns:
        df = df.withColumn("hour", F.lit(0))
    if "is_weekend" not in df.columns:
        df = df.withColumn("is_weekend", F.lit(0))

    if surge_model is None:
        demand_factor = F.least(F.greatest(F.col("predicted_demand") / F.lit(60.0), F.lit(0.5)), F.lit(2.0))
        supply_factor = F.lit(1.0) + F.greatest(F.lit(1.0) - F.col("supply_index"), F.lit(0.0)) * F.lit(0.8)
        rain_factor = F.lit(1.0) + F.least(F.greatest(F.col("rain_intensity"), F.lit(0.0)), F.lit(1.0)) * F.lit(0.25)
        raw = F.lit(1.0) * demand_factor * supply_factor * rain_factor
        return df.withColumn("predicted_surge_multiplier", F.round(F.least(F.greatest(raw, F.lit(1.0)), F.lit(3.0)), 3))

    feature_cols = [
        "predicted_demand",
        "supply_index",
        "rain_intensity",
        "hour",
        "is_weekend",
    ]

    @pandas_udf(DoubleType())
    def predict_surge_udf(*cols: pd.Series) -> pd.Series:
        X = pd.concat(cols, axis=1)
        X.columns = feature_cols
        preds = surge_model.predict(X)
        preds = pd.Series(preds).astype(float).clip(lower=1.0, upper=3.0)
        return preds

    return df.withColumn("predicted_surge_multiplier", F.round(predict_surge_udf(*[F.col(c) for c in feature_cols]), 3))

def add_run_metadata(df: DataFrame, run_id: str) -> DataFrame:
    return (
        df.withColumn("scoring_run_id", F.lit(run_id))
        .withColumn("scored_at_utc", F.lit(datetime.utcnow().isoformat()))
    )

def pick_primary_key_cols(df: DataFrame) -> Tuple[str, ...]:
    if "event_id" in df.columns:
        return ("event_id",)
    if all(c in df.columns for c in ["city", "zone_name", "event_ts"]):
        return ("city", "zone_name", "event_ts")
    if all(c in df.columns for c in ["city", "zone_name", "hour", "event_date"]):
        return ("city", "zone_name", "hour", "event_date")
    # last-resort key (not ideal but keeps pipeline functional)
    return ("city", "event_date")

def write_predictions_merge(df: DataFrame, out_path: str) -> None:
    df = ensure_partition_cols(df)
    pk_cols = pick_primary_key_cols(df)

    merge_cond_parts = [f"t.{c} = s.{c}" for c in pk_cols if c in df.columns]
    merge_cond_parts.append("t.scoring_run_id = s.scoring_run_id")
    merge_condition = " AND ".join(merge_cond_parts)

    if not DeltaTable.isDeltaTable(df.sparkSession, out_path):
        (
            df.write.format("delta")
            .mode("overwrite")
            .partitionBy("event_date", "city")
            .save(out_path)
        )
        return

    delta_tbl = DeltaTable.forPath(df.sparkSession, out_path)

    (
        delta_tbl.alias("t")
        .merge(df.alias("s"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

def compute_error_metrics(df: DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    # Demand metrics
    demand_label = "actual_demand" if "actual_demand" in df.columns else ("demand" if "demand" in df.columns else None)
    if demand_label and "predicted_demand" in df.columns:
        d = df.select(F.col(demand_label).cast("double").alias("y"), F.col("predicted_demand").cast("double").alias("yhat")).dropna()
        if d.take(1):
            agg = d.select(
                F.avg(F.abs(F.col("y") - F.col("yhat"))).alias("mae"),
                F.sqrt(F.avg(F.pow(F.col("y") - F.col("yhat"), 2))).alias("rmse"),
                F.avg(
                    F.when(F.col("y") != 0, F.abs((F.col("y") - F.col("yhat")) / F.col("y"))).otherwise(None)
                ).alias("mape"),
            ).collect()[0]
            metrics["demand_mae"] = float(agg["mae"]) if agg["mae"] is not None else None
            metrics["demand_rmse"] = float(agg["rmse"]) if agg["rmse"] is not None else None
            metrics["demand_mape"] = float(agg["mape"]) if agg["mape"] is not None else None

    # Surge metrics
    surge_label = (
        "actual_surge_multiplier"
        if "actual_surge_multiplier" in df.columns
        else ("surge_multiplier" if "surge_multiplier" in df.columns else None)
    )
    if surge_label and "predicted_surge_multiplier" in df.columns:
        s = df.select(F.col(surge_label).cast("double").alias("y"), F.col("predicted_surge_multiplier").cast("double").alias("yhat")).dropna()
        if s.take(1):
            agg = s.select(
                F.avg(F.abs(F.col("y") - F.col("yhat"))).alias("mae"),
                F.sqrt(F.avg(F.pow(F.col("y") - F.col("yhat"), 2))).alias("rmse"),
                F.avg(
                    F.when(F.col("y") != 0, F.abs((F.col("y") - F.col("yhat")) / F.col("y"))).otherwise(None)
                ).alias("mape"),
            ).collect()[0]
            metrics["surge_mae"] = float(agg["mae"]) if agg["mae"] is not None else None
            metrics["surge_rmse"] = float(agg["rmse"]) if agg["rmse"] is not None else None
            metrics["surge_mape"] = float(agg["mape"]) if agg["mape"] is not None else None

    return metrics

def write_metrics(spark: SparkSession, metrics_out: str, run_id: str, df: DataFrame) -> None:
    total = df.count()
    null_demand = df.filter(F.col("predicted_demand").isNull()).count() if "predicted_demand" in df.columns else None
    null_surge = df.filter(F.col("predicted_surge_multiplier").isNull()).count() if "predicted_surge_multiplier" in df.columns else None

    extra = compute_error_metrics(df)

    row = {
        "scoring_run_id": run_id,
        "scored_at_utc": datetime.utcnow().isoformat(),
        "total_rows": total,
        "null_predicted_demand": null_demand,
        "null_predicted_surge_multiplier": null_surge,
        **extra,
    }

    spark.createDataFrame([row]).write.format("delta").mode("append").save(metrics_out)

def main():
    defaults = default_paths()

    parser = argparse.ArgumentParser(description="Batch scoring for mobility demand and surge models")
    parser.add_argument("--silver_features", default=os.getenv("SILVER_FEATURES_PATH", defaults["silver_features"]))
    parser.add_argument("--predictions_out", default=os.getenv("BATCH_PREDICTIONS_OUT", defaults["predictions_out"]))
    parser.add_argument("--metrics_out", default=os.getenv("BATCH_METRICS_OUT", defaults["metrics_out"]))
    parser.add_argument("--demand_model", default=os.getenv("DEMAND_MODEL_PATH", defaults["demand_model"]))
    parser.add_argument("--surge_model", default=os.getenv("SURGE_MODEL_PATH", defaults["surge_model"]))
    args = parser.parse_args()

    run_id = os.getenv("SCORING_RUN_ID", f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[BatchScoring] run_id={run_id}")
    print(f"[BatchScoring] silver_features={args.silver_features}")
    print(f"[BatchScoring] predictions_out={args.predictions_out}")
    print(f"[BatchScoring] metrics_out={args.metrics_out}")

    demand_model = load_local_model(args.demand_model)
    surge_model = load_local_model(args.surge_model)

    df = read_silver_features(spark, args.silver_features)
    df = ensure_partition_cols(df)

    df = score_demand(df, demand_model)
    df = score_surge(df, surge_model)
    df = add_run_metadata(df, run_id)

    # merge write
    write_predictions_merge(df, args.predictions_out)
    # metrics write
    write_metrics(spark, args.metrics_out, run_id, df)
    print("[BatchScoring] Done")

if __name__ == "__main__":
    main()
    