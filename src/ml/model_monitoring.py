import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import math
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def build_spark(app_name: str = "mobility-model-monitoring") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

def default_paths() -> Dict[str, str]:
    root = repo_root()
    return {
        "predictions_delta": str(root / "data" / "gold" / "batch_predictions"),
        "monitoring_out": str(root / "data" / "gold" / "model_monitoring_metrics"),
        "api_logs_dir": str(root / "data" / "api_logs"),
        "drift_out": str(root / "data" / "gold" / "model_drift_details"),
        "config_yaml": str(root / "config" / "config.yaml"),
    }

def load_yaml_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def deep_get(d: Dict[str, Any], keys: List[str], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def compute_regression_metrics(df: DataFrame, y_col: str, yhat_col: str) -> Dict[str, Optional[float]]:
    d = df.select(F.col(y_col).cast("double").alias("y"), F.col(yhat_col).cast("double").alias("yhat")).dropna()
    if not d.take(1):
        return {"mae": None, "rmse": None, "mape": None}

    agg = d.select(
        F.avg(F.abs(F.col("y") - F.col("yhat"))).alias("mae"),
        F.sqrt(F.avg(F.pow(F.col("y") - F.col("yhat"), 2))).alias("rmse"),
        F.avg(
            F.when(F.col("y") != 0, F.abs((F.col("y") - F.col("yhat")) / F.col("y"))).otherwise(None)
        ).alias("mape"),
    ).collect()[0]

    return {
        "mae": float(agg["mae"]) if agg["mae"] is not None else None,
        "rmse": float(agg["rmse"]) if agg["rmse"] is not None else None,
        "mape": float(agg["mape"]) if agg["mape"] is not None else None,
    }

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

# Drift detection (PSI)
def _psi_from_hist(base: List[int], cur: List[int], eps: float = 1e-6) -> float:
    base_total = sum(base)
    cur_total = sum(cur)
    if base_total == 0 or cur_total == 0:
        return 0.0

    psi = 0.0
    for b, c in zip(base, cur):
        b_pct = max(b / base_total, eps)
        c_pct = max(c / cur_total, eps)
        psi += (c_pct - b_pct) * math.log(c_pct / b_pct)
    return float(psi)

def psi_for_numeric_feature(df_base: DataFrame, df_cur: DataFrame, col: str, bins: int = 10) -> Optional[float]:
    if col not in df_base.columns or col not in df_cur.columns:
        return None

    base_nonnull = df_base.select(F.col(col).cast("double").alias(col)).dropna()
    cur_nonnull = df_cur.select(F.col(col).cast("double").alias(col)).dropna()

    if not base_nonnull.take(1) or not cur_nonnull.take(1):
        return None

    # quantile split points (bins-1 cutpoints)
    probs = [i / bins for i in range(1, bins)]
    cuts = base_nonnull.approxQuantile(col, probs, 0.01)

    def bucketize(df: DataFrame) -> DataFrame:
        expr = F.when(F.col(col) <= cuts[0], F.lit(0))
        for i in range(1, len(cuts)):
            expr = expr.when((F.col(col) > cuts[i - 1]) & (F.col(col) <= cuts[i]), F.lit(i))
        expr = expr.otherwise(F.lit(len(cuts)))
        return df.select(expr.alias("bucket"))

    base_hist = bucketize(base_nonnull).groupBy("bucket").count().orderBy("bucket").collect()
    cur_hist = bucketize(cur_nonnull).groupBy("bucket").count().orderBy("bucket").collect()

    # ensure full bin coverage
    base_counts = [0] * (bins)
    cur_counts = [0] * (bins)

    for r in base_hist:
        base_counts[int(r["bucket"])] = int(r["count"])
    for r in cur_hist:
        cur_counts[int(r["bucket"])] = int(r["count"])

    return _psi_from_hist(base_counts, cur_counts)

def missing_rate(df: DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    total = df.count()
    if total == 0:
        return None
    miss = df.filter(F.col(col).isNull()).count()
    return float(miss / total)

# API latency monitoring (JSONL)
def read_api_jsonl_logs(api_logs_dir: str, days: int = 1) -> List[Dict[str, Any]]:
    root = Path(api_logs_dir)
    out: List[Dict[str, Any]] = []
    for i in range(days):
        day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        p = root / f"predictions_{day}.jsonl"
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                out.append(json.loads(line))
        except Exception:
            # skip bad file
            continue
    return out

def latency_stats(events: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    latencies = [safe_float(e.get("latency_ms")) for e in events]
    latencies = [x for x in latencies if x is not None]
    if not latencies:
        return {"p50_ms": None, "p95_ms": None, "avg_ms": None}

    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> float:
        idx = min(max(int(round(p * (n - 1))), 0), n - 1)
        return float(latencies[idx])

    return {
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "avg_ms": float(sum(latencies) / n),
    }

# Alerting hooks
def emit_alert(message: str) -> None:
    print(f"[ALERT] {message}")

    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        return

    try:
        import requests
        requests.post(webhook, json={"text": message}, timeout=5)
    except Exception:
        print("[ALERT] Slack webhook failed (ignored)")

# Main monitoring
def filter_window(df: DataFrame, date_col: str, start_date: str, end_date: str) -> DataFrame:
    if date_col not in df.columns:
        return df
    return df.filter((F.col(date_col) >= F.lit(start_date)) & (F.col(date_col) < F.lit(end_date)))

def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Model monitoring job")
    parser.add_argument("--predictions_delta", default=os.getenv("PREDICTIONS_DELTA_PATH", defaults["predictions_delta"]))
    parser.add_argument("--monitoring_out", default=os.getenv("MONITORING_OUT_PATH", defaults["monitoring_out"]))
    parser.add_argument("--api_logs_dir", default=os.getenv("API_LOGS_DIR", defaults["api_logs_dir"]))
    parser.add_argument("--baseline_days", type=int, default=int(os.getenv("BASELINE_DAYS", "14")))
    parser.add_argument("--current_days", type=int, default=int(os.getenv("CURRENT_DAYS", "1")))
    parser.add_argument("--drift_out", default=os.getenv("DRIFT_OUT_PATH", defaults["drift_out"]))
    parser.add_argument("--config_yaml", default=os.getenv("CONFIG_YAML_PATH", defaults["config_yaml"]))

    # Drift thresholds
    parser.add_argument("--psi_warn", type=float, default=float(os.getenv("PSI_WARN", "0.1")))
    parser.add_argument("--psi_crit", type=float, default=float(os.getenv("PSI_CRIT", "0.2")))
    parser.add_argument("--missing_shift_warn", type=float, default=float(os.getenv("MISSING_SHIFT_WARN", "0.05")))

    args = parser.parse_args()

    cfg = load_yaml_config(args.config_yaml)

    args.psi_warn = float(os.getenv("PSI_WARN", str(deep_get(cfg, ["monitoring", "psi_warn"], args.psi_warn))))
    args.psi_crit = float(os.getenv("PSI_CRIT", str(deep_get(cfg, ["monitoring", "psi_crit"], args.psi_crit))))
    args.missing_shift_warn = float(os.getenv("MISSING_SHIFT_WARN", str(deep_get(cfg, ["monitoring", "missing_shift_warn"], args.missing_shift_warn))))

    volume_drop_ratio = float(os.getenv("VOLUME_DROP_RATIO", str(deep_get(cfg, ["monitoring", "volume_drop_ratio"], 0.3))))

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    today = datetime.utcnow().date()
    cur_end = today.strftime("%Y-%m-%d")
    cur_start = (today - timedelta(days=args.current_days)).strftime("%Y-%m-%d")

    base_end = cur_start
    base_start = (today - timedelta(days=args.baseline_days + args.current_days)).strftime("%Y-%m-%d")

    run_id = os.getenv("MONITORING_RUN_ID", f"mon_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    print(f"[Monitoring] run_id={run_id}")
    print(f"[Monitoring] baseline={base_start} -> {base_end}")
    print(f"[Monitoring] current ={cur_start} -> {cur_end}")

    df_all = spark.read.format("delta").load(args.predictions_delta)

    date_col = "event_date" if "event_date" in df_all.columns else None
    if date_col is None:
        date_col = df_all.columns[0]

    df_base = filter_window(df_all, "event_date", base_start, base_end) if "event_date" in df_all.columns else df_all
    df_cur = filter_window(df_all, "event_date", cur_start, cur_end) if "event_date" in df_all.columns else df_all

    # Volume monitoring
    base_count = df_base.count()
    cur_count = df_cur.count()

    # Performance metrics 
    demand_label = "actual_demand" if "actual_demand" in df_all.columns else ("demand" if "demand" in df_all.columns else None)
    surge_label = (
        "actual_surge_multiplier"
        if "actual_surge_multiplier" in df_all.columns
        else ("surge_multiplier" if "surge_multiplier" in df_all.columns else None)
    )

    demand_perf = compute_regression_metrics(df_cur, demand_label, "predicted_demand") if demand_label and "predicted_demand" in df_all.columns else {"mae": None, "rmse": None, "mape": None}
    surge_perf = compute_regression_metrics(df_cur, surge_label, "predicted_surge_multiplier") if surge_label and "predicted_surge_multiplier" in df_all.columns else {"mae": None, "rmse": None, "mape": None}

    # Drift checks 
    drift_features = [c for c in ["demand_avg_7d", "avg_surge", "avg_fare", "supply_index", "rain_intensity"] if c in df_all.columns]
    drift_results: Dict[str, Any] = {}

    for col in drift_features:
        psi = psi_for_numeric_feature(df_base, df_cur, col, bins=10)
        base_miss = missing_rate(df_base, col)
        cur_miss = missing_rate(df_cur, col)

        drift_results[col] = {
            "psi": psi,
            "baseline_missing_rate": base_miss,
            "current_missing_rate": cur_miss,
            "missing_rate_shift": (cur_miss - base_miss) if (cur_miss is not None and base_miss is not None) else None,
        }

        # Alerts
        if psi is not None:
            if psi >= args.psi_crit:
                emit_alert(f"CRITICAL drift detected for {col}: PSI={psi:.3f}")
            elif psi >= args.psi_warn:
                emit_alert(f"Drift warning for {col}: PSI={psi:.3f}")

        shift = drift_results[col]["missing_rate_shift"]
        if shift is not None and shift >= args.missing_shift_warn:
            emit_alert(f"Missing rate shift for {col}: shift={shift:.3f}")

    # Write drift details as a separate Delta table (one row per feature)
    drift_rows = []
    for feature, vals in drift_results.items():
        drift_rows.append({
            "monitoring_run_id": run_id,
            "generated_at_utc": datetime.utcnow().isoformat(),
            "baseline_start": base_start,
            "baseline_end": base_end,
            "current_start": cur_start,
            "current_end": cur_end,
            "feature": feature,
            "psi": vals.get("psi"),
            "baseline_missing_rate": vals.get("baseline_missing_rate"),
            "current_missing_rate": vals.get("current_missing_rate"),
            "missing_rate_shift": vals.get("missing_rate_shift"),
        })

    if drift_rows:
        spark.createDataFrame(drift_rows).write.format("delta").mode("append").save(args.drift_out)

    # Latency checks 
    events = read_api_jsonl_logs(args.api_logs_dir, days=args.current_days)
    lat = latency_stats(events)

    # Prediction volume alerts 
    if base_count > 0 and cur_count < max(1, int(base_count * volume_drop_ratio)):
        emit_alert(f"Prediction volume drop: baseline={base_count}, current={cur_count}")

    record = {
        "monitoring_run_id": run_id,
        "generated_at_utc": datetime.utcnow().isoformat(),
        "baseline_start": base_start,
        "baseline_end": base_end,
        "current_start": cur_start,
        "current_end": cur_end,
        "baseline_rows": base_count,
        "current_rows": cur_count,
        "demand_mae": demand_perf["mae"],
        "demand_rmse": demand_perf["rmse"],
        "demand_mape": demand_perf["mape"],
        "surge_mae": surge_perf["mae"],
        "surge_rmse": surge_perf["rmse"],
        "surge_mape": surge_perf["mape"],
        "latency_p50_ms": lat["p50_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "latency_avg_ms": lat["avg_ms"],
        "drift_json": json.dumps(drift_results),
    }

    spark.createDataFrame([record]).write.format("delta").mode("append").save(args.monitoring_out)

    print("[Monitoring] Wrote monitoring metrics")
    print("[Monitoring] Done")

if __name__ == "__main__":
    main()
