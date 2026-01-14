import sys

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lag, lead, avg, sum as spark_sum, count, max as spark_max,
    stddev, when, lit, datediff, hour, dayofweek, 
    countDistinct as f_countDistinct, col, date_trunc, count as f_count,
    to_date, hour, avg as f_avg )
from pyspark.sql.window import Window
from src.utils.spark_session import get_spark_session
from src.utils.delta_utils import read_delta, write_delta_overwrite, ensure_path
from src.utils.logger import setup_logger
from typing import Optional
import yaml

logger = setup_logger(__name__, "logs")

class DeltaLakeManager:
    def __init__(self, spark):
        self.spark = spark

    def read_delta(self, path: str) -> DataFrame:
        return read_delta(self.spark, path)

    def write_delta(
        self,
        df: DataFrame,
        path: str,
        mode: str = "overwrite",
        partition_by: Optional[list[str]] = None,
    ) -> None:
        if mode != "overwrite":
            raise ValueError(f"Only mode='overwrite' is supported right now, got: {mode}")
        ensure_path(path)
        write_delta_overwrite(df, path, partition_by=partition_by)

# ML features from gold layer data
class FeatureEngineer:
    def __init__(self, config_path='config/config.yaml'):
        logger.info("Initializing Feature Engineer...")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.spark = get_spark_session(config_path,
                                        "local[*]",
                                        8,
                                        "UTC",)
        self.delta_manager = DeltaLakeManager(self.spark)
    
    def create_demand_features(self):
        
        logger.info("Creating Demand Forecasting Features")
        
        # Hourly demand data
        trips_fact = self.delta_manager.read_delta("./data/gold_delta/trips_fact")

        hourly_surge = (
            trips_fact
            .filter(col("requested_at").isNotNull())
            .withColumn("hour_ts", date_trunc("hour", col("requested_at")))
            .groupBy("hour_ts", "pickup_location_id")
            .agg(
                f_avg(col("surge_multiplier")).alias("avg_surge")
            )
        )
        
        hourly_demand = (
            trips_fact
            .filter(col("requested_at").isNotNull())
            .withColumn("hour_ts", date_trunc("hour", col("requested_at")))
            .groupBy("hour_ts", "pickup_location_id")
            .agg(f_count("*").alias("trip_count"))
        )

        hourly_demand = (
            hourly_demand
            .withColumn("trip_date", to_date(col("hour_ts")))
            .withColumn("hour", hour(col("hour_ts")))
        )

        hourly_demand = hourly_demand.cache()

        demand_features = hourly_demand.join(
            hourly_surge,
            on=["hour_ts", "pickup_location_id"],
            how="left",
        )        

        # Window for time-based features
        window_spec = Window.partitionBy("pickup_location_id", "hour").orderBy('hour_ts')
        
        # Create lag features 
        demand_features = demand_features \
            .withColumn('demand_lag_1', lag('trip_count', 1).over(window_spec)) \
            .withColumn('demand_lag_7', lag('trip_count', 7).over(window_spec)) \
            .withColumn('demand_lag_14', lag('trip_count', 14).over(window_spec))

        demand_features = demand_features \
            .withColumn("avg_surge_lag_1", lag("avg_surge", 1).over(window_spec))
        
        # Rolling averages
        window_7d = Window.partitionBy("pickup_location_id", "hour") \
            .orderBy("hour_ts") \
            .rowsBetween(-168, -1)

        window_30d = Window.partitionBy("pickup_location_id", "hour") \
            .orderBy("hour_ts") \
            .rowsBetween(-720, -1)
        
        demand_features = demand_features \
            .withColumn('demand_avg_7d', avg('trip_count').over(window_7d)) \
            .withColumn('demand_avg_30d', avg('trip_count').over(window_30d)) \
            .withColumn('demand_std_7d', stddev('trip_count').over(window_7d))
        
        # Time-based features
        demand_features = demand_features \
            .withColumn('is_peak_hour', 
                when((col('hour') >= 7) & (col('hour') <= 9), 1)
                .when((col('hour') >= 17) & (col('hour') <= 19), 1)
                .otherwise(0)
            ) \
            .withColumn('is_night', 
                when((col('hour') >= 22) | (col('hour') <= 5), 1).otherwise(0)
            ) \
            .withColumn('is_business_hours',
                when((col('hour') >= 9) & (col('hour') <= 17), 1).otherwise(0)
            )
        
        # Remove nulls from lag features
        demand_features = demand_features.na.drop()
        
        # Select final feature columns
        feature_cols = [
            "trip_date",
            "hour",
            "pickup_location_id",

            "trip_count",

            "demand_lag_1",
            "demand_lag_7",
            "demand_lag_14",

            "demand_avg_7d",
            "demand_avg_30d",
            "demand_std_7d",

            "is_peak_hour",
            "is_night",
            "is_business_hours",

            "avg_surge",
            "avg_surge_lag_1",
        ]
        
        demand_features = demand_features.select(*feature_cols)
        
        # Save features
        self.delta_manager.write_delta(
            demand_features,
            './data/features/demand_forecasting',
            mode='overwrite'
        )
        
        logger.info(f"Demand features created: {demand_features.count()} records")
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return demand_features
    
    def create_surge_pricing_features(self):
        logger.info("Creating Surge Pricing Features")

        trips = self.delta_manager.read_delta("./data/gold_delta/trips_fact")

        # Build an hourly grain from trips_fact
        base = (
            trips
            .filter(col("requested_at").isNotNull())
            .withColumn("hour_ts", date_trunc("hour", col("requested_at")))
            .withColumn("trip_date", to_date(col("hour_ts")))
            .withColumn("hour", hour(col("hour_ts")))
        )

        surge_agg = (
            base
            .filter(col("status") == "completed")
            .groupBy("hour_ts", "trip_date", "hour", "pickup_location_id")
            .agg(
                f_count("*").alias("trip_count"),
                f_countDistinct("driver_key").alias("driver_count"),
                f_avg(col("surge_multiplier")).alias("avg_surge"),
                spark_max(col("surge_multiplier")).alias("max_surge"),
                f_avg(col("fare_amount")).alias("avg_fare"),
                f_avg(col("distance_km")).alias("avg_distance"),
                f_avg(col("duration_min")).alias("avg_duration"),
            )
            .withColumn(
                "demand_supply_ratio",
                when(col("driver_count") == 0, lit(None)).otherwise(col("trip_count") / col("driver_count"))
            )
            # day_of_week
            .withColumn("day_of_week", dayofweek(col("trip_date")))
            .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
        )

        surge_agg = surge_agg.cache()

        # Time-based features
        surge_features = (
            surge_agg
            .withColumn(
                "is_peak_hour",
                when((col("hour") >= 7) & (col("hour") <= 9), 1)
                .when((col("hour") >= 17) & (col("hour") <= 19), 1)
                .otherwise(0)
            )
            .withColumn("is_late_night", when((col("hour") >= 23) | (col("hour") <= 3), 1).otherwise(0))
        )

        # Lag + rolling windows 
        window_spec = Window.partitionBy("pickup_location_id", "hour").orderBy("hour_ts")

        surge_features = (
            surge_features
            .withColumn("surge_lag_1", lag("avg_surge", 1).over(window_spec))
            .withColumn("demand_supply_lag_1", lag("demand_supply_ratio", 1).over(window_spec))
        )

        window_7d = (
            Window.partitionBy("pickup_location_id", "hour")
            .orderBy("hour_ts")
            .rowsBetween(-168, -1)  
        )

        surge_features = (
            surge_features
            .withColumn("surge_avg_7d", avg("avg_surge").over(window_7d))
            .withColumn("demand_supply_avg_7d", avg("demand_supply_ratio").over(window_7d))
            .withColumn("trip_count_avg_7d", avg("trip_count").over(window_7d))
            .withColumn(
                "likely_bad_weather",
                when((col("trip_count") < col("trip_count_avg_7d") * 0.7) & (col("avg_surge") > 1.3), 1).otherwise(0)
            )
        )

        surge_features = surge_features.na.drop(subset=["surge_lag_1", "demand_supply_lag_1"])

        feature_cols = [
            "trip_date", "hour", "day_of_week", "is_weekend",
            "pickup_location_id",
            "avg_surge",
            "max_surge",
            "trip_count", "driver_count", "demand_supply_ratio",
            "avg_fare", "avg_distance", "avg_duration",
            "is_peak_hour", "is_late_night",
            "surge_lag_1", "demand_supply_lag_1",
            "surge_avg_7d", "demand_supply_avg_7d", "trip_count_avg_7d",
            "likely_bad_weather",
        ]

        surge_features = surge_features.select(*feature_cols)

        self.delta_manager.write_delta(
            surge_features,
            "./data/features/surge_pricing",
            mode="overwrite"
        )

        logger.info(f"Surge pricing features created: {surge_features.count()} records")
        logger.info(f"Feature columns: {len(feature_cols)}")

        return surge_features
    
    def create_driver_churn_features(self):
        logger.info("Creating Driver Churn Features")

        trips = self.delta_manager.read_delta("./data/gold_delta/trips_fact")

        base = (
            trips
            .filter(col("requested_at").isNotNull())
            .withColumn("trip_date", to_date(col("requested_at")))
        )

        # Daily activity per driver
        driver_daily = (
            base
            .filter(col("status") == "completed")
            .groupBy("driver_key", "trip_date")
            .agg(
                count("*").alias("daily_trips"),
                spark_sum(col("fare_amount")).alias("daily_earnings"),
                f_avg(col("duration_min")).alias("daily_avg_duration_min"),
                spark_sum(col("duration_min")).alias("daily_minutes_worked"),
            )
        )

        w = Window.partitionBy("driver_key").orderBy("trip_date")

        churn_features = (
            driver_daily
            .withColumn("trips_lag_7", lag("daily_trips", 7).over(w))
            .withColumn("trips_lag_14", lag("daily_trips", 14).over(w))
            .withColumn("trips_lag_30", lag("daily_trips", 30).over(w))
        )

        w30 = Window.partitionBy("driver_key").orderBy("trip_date").rowsBetween(-30, -1)

        churn_features = (
            churn_features
            .withColumn("avg_trips_30d", avg("daily_trips").over(w30))
            .withColumn("avg_earnings_30d", avg("daily_earnings").over(w30))
            .withColumn("avg_minutes_30d", avg("daily_minutes_worked").over(w30))
            .withColumn("trips_declining", when(col("daily_trips") < col("avg_trips_30d") * 0.5, 1).otherwise(0))
            .withColumn("earnings_declining", when(col("daily_earnings") < col("avg_earnings_30d") * 0.5, 1).otherwise(0))
        )

        churn_features = churn_features.withColumn(
            "next_trip_date",
            lead("trip_date", 1).over(w)
        ).withColumn(
            "days_to_next_trip",
            datediff(col("next_trip_date"), col("trip_date"))
        ).withColumn(
            "will_churn",
            when(col("next_trip_date").isNull() | (col("days_to_next_trip") > 30), 1).otherwise(0)
        )

        churn_features = churn_features.na.drop()

        feature_cols = [
            "driver_key", "trip_date",
            "daily_trips", "daily_earnings", "daily_avg_duration_min", "daily_minutes_worked",
            "trips_lag_7", "trips_lag_14", "trips_lag_30",
            "avg_trips_30d", "avg_earnings_30d", "avg_minutes_30d",
            "trips_declining", "earnings_declining",
            "will_churn",
        ]

        churn_features = churn_features.select(*feature_cols)

        self.delta_manager.write_delta(
            churn_features,
            "./data/features/driver_churn",
            mode="overwrite"
        )

        logger.info(f"Driver churn features created: {churn_features.count()} records")
        logger.info(f"Feature columns: {len(feature_cols)}")

        return churn_features
    
    def create_all_features(self):
        logger.info("Creating All ML Features")
        
        try:
            # Create features for each use case
            demand_features = self.create_demand_features()
            surge_features = self.create_surge_pricing_features()
            churn_features = self.create_driver_churn_features()
            
            logger.info("All ML Features Created Successfully!")
            logger.info("\nFeature Tables:")
            logger.info(f"  - Demand Forecasting: {demand_features.count():,} records")
            logger.info(f"  - Surge Pricing: {surge_features.count():,} records")
            logger.info(f"  - Driver Churn: {churn_features.count():,} records")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

def main(): 
    engineer = FeatureEngineer()
    engineer.create_all_features()

if __name__ == "__main__":
    main()