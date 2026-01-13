from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.external_task import ExternalTaskSensor
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.transformation.silver_to_gold import GoldLayerTransformer
from src.utils.logger import setup_logger

logger = setup_logger(__name__,"logs")


default_args = {
    'owner': 'data-team',
    'depends_on_past': True,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def create_all_dimensions(**context):
    logger.info("Creating dimension tables...")
    
    transformer = GoldLayerTransformer()
    
    transformer.create_dim_users()
    transformer.create_dim_drivers()
    transformer.create_dim_vehicles()
    transformer.create_dim_locations()
    transformer.create_dim_time()
    
    logger.info("All dimension tables created")


def create_trips_fact_table(**context):
    logger.info("Creating trips_fact table...")
    
    transformer = GoldLayerTransformer()
    transformer.create_trips_fact()
    
    logger.info("trips_fact created")


def create_payments_fact_table(**context):
    logger.info("Creating payments_fact table...")
    
    transformer = GoldLayerTransformer()
    transformer.create_payments_fact()
    
    logger.info("payments_fact created")


def create_daily_aggregates(**context):
    logger.info("Creating daily metrics...")
    
    transformer = GoldLayerTransformer()
    transformer.create_daily_metrics()
    
    logger.info("daily_metrics created")


def create_driver_aggregates(**context):
    logger.info("Creating driver performance metrics...")
    
    transformer = GoldLayerTransformer()
    transformer.create_driver_performance()
    
    logger.info("driver_performance created")


def create_hourly_demand(**context):
    from pyspark.sql.functions import hour, col, count, avg, sum as spark_sum
    from src.utils.spark_session import get_spark_session
    from src.utils.delta_utils import DeltaLakeManager
    
    logger.info("Creating hourly demand aggregation...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    
    # Read trips_fact
    trips = delta.read_delta('./data/gold_delta/trips_fact')
    dim_time = delta.read_delta('./data/gold_delta/dim_time')
    dim_locations = delta.read_delta('./data/gold_delta/dim_locations')
    
    # Join with time and location
    hourly_demand = trips \
        .filter(col('trip_status') == 'completed') \
        .join(dim_time, trips.start_time_id == dim_time.time_id) \
        .join(dim_locations, trips.start_location_id == dim_locations.location_id) \
        .groupBy(
            'trip_date',
            dim_time.hour,
            dim_time.day_of_week,
            dim_time.is_weekend,
            dim_locations.city,
            dim_locations.zone_name
        ) \
        .agg(
            count('trip_id').alias('trip_count'),
            avg('total_amount').alias('avg_fare'),
            avg('surge_multiplier').alias('avg_surge'),
            spark_sum('distance_km').alias('total_distance')
        )
    
    # Save for ML feature engineering
    delta.write_delta(
        hourly_demand,
        './data/gold_delta/hourly_demand',
        mode='overwrite'
    )
    
    logger.info(f"hourly_demand created: {hourly_demand.count()} records")


def create_location_metrics(**context):
    from pyspark.sql.functions import col, count, avg, sum as spark_sum
    from src.utils.spark_session import get_spark_session
    from src.utils.delta_utils import DeltaLakeManager
    
    logger.info("Creating location metrics...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    
    trips = delta.read_delta('./data/gold_delta/trips_fact')
    dim_locations = delta.read_delta('./data/gold_delta/dim_locations')
    
    location_metrics = trips \
        .filter(col('trip_status') == 'completed') \
        .join(dim_locations, trips.start_location_id == dim_locations.location_id) \
        .groupBy(
            dim_locations.location_id,
            dim_locations.city,
            dim_locations.zone_name,
            dim_locations.zone_type
        ) \
        .agg(
            count('trip_id').alias('total_trips'),
            avg('total_amount').alias('avg_fare'),
            avg('distance_km').alias('avg_distance'),
            avg('duration_minutes').alias('avg_duration'),
            avg('surge_multiplier').alias('avg_surge')
        ) \
        .orderBy(col('total_trips').desc())
    
    delta.write_delta(
        location_metrics,
        './data/gold_delta/location_metrics',
        mode='overwrite'
    )
    
    logger.info(f"location_metrics created: {location_metrics.count()} records")


def validate_star_schema(**context):
    from src.utils.spark_session import get_spark_session
    from src.utils.delta_utils import DeltaLakeManager
    from src.utils.data_quality import DataQualityChecker
    
    logger.info("Validating star schema...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    quality = DataQualityChecker(spark)
    
    # Check referential integrity
    trips = delta.read_delta('./data/gold_delta/trips_fact')
    dim_users = delta.read_delta('./data/gold_delta/dim_users')
    dim_drivers = delta.read_delta('./data/gold_delta/dim_drivers')
    
    # Validate foreign keys
    orphan_users = quality.check_referential_integrity(
        trips, dim_users, 'user_id', 'user_id'
    )
    
    orphan_drivers = quality.check_referential_integrity(
        trips, dim_drivers, 'driver_id', 'driver_id'
    )
    
    if orphan_users > 0 or orphan_drivers > 0:
        logger.warning(f"Referential integrity issues found!")
    else:
        logger.info("Star schema validation passed")


def optimize_gold_tables(**context):
    from src.utils.spark_session import get_spark_session
    from src.utils.delta_utils import DeltaLakeManager
    
    logger.info("Optimizing Gold layer tables...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    
    # Optimize fact tables with Z-ordering
    delta.optimize_table('./data/gold_delta/trips_fact', zorder_by=['trip_date', 'user_id'])
    delta.optimize_table('./data/gold_delta/payments_fact', zorder_by=['user_id'])
    
    # Optimize dimension tables
    delta.optimize_table('./data/gold_delta/dim_users')
    delta.optimize_table('./data/gold_delta/dim_drivers')
    delta.optimize_table('./data/gold_delta/dim_time')
    
    logger.info("Gold layer optimization complete")


def check_gold_completeness(**context):
    from src.utils.spark_session import get_spark_session
    from src.utils.delta_utils import DeltaLakeManager
    
    logger.info("Checking Gold layer completeness...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    
    required_tables = [
        'dim_users', 'dim_drivers', 'dim_vehicles', 'dim_locations', 'dim_time',
        'trips_fact', 'payments_fact',
        'daily_metrics', 'driver_performance',
        'hourly_demand', 'location_metrics'
    ]
    
    missing_tables = []
    
    for table in required_tables:
        path = f'./data/gold_delta/{table}'
        if not delta.create_table_if_not_exists(path):
            missing_tables.append(table)
    
    if missing_tables:
        logger.error(f"Missing tables: {missing_tables}")
        raise ValueError(f"Missing Gold tables: {missing_tables}")
    
    logger.info("All Gold layer tables present")


# Create DAG
with DAG(
    'gold_layer_aggregation',
    default_args=default_args,
    description='Create Gold layer star schema and aggregations',
    schedule_interval='0 3 * * *',  # Daily at 3 AM (after Silver)
    catchup=False,
    tags=['gold', 'star-schema', 'daily'],
) as dag:
    
    # Wait for Silver DAG
    wait_for_silver = ExternalTaskSensor(
        task_id='wait_for_silver',
        external_dag_id='silver_layer_transformation',
        external_task_id='end',
        timeout=600,
        mode='poke',
        poke_interval=60
    )
    
    start = DummyOperator(task_id='start')
    
    # Create dimensions
    create_dims = PythonOperator(
        task_id='create_all_dimensions',
        python_callable=create_all_dimensions,
        provide_context=True
    )
    
    # Create facts (parallel)
    create_trips = PythonOperator(
        task_id='create_trips_fact',
        python_callable=create_trips_fact_table,
        provide_context=True
    )
    
    create_payments = PythonOperator(
        task_id='create_payments_fact',
        python_callable=create_payments_fact_table,
        provide_context=True
    )
    
    # Create aggregates (parallel)
    create_daily = PythonOperator(
        task_id='create_daily_aggregates',
        python_callable=create_daily_aggregates,
        provide_context=True
    )
    
    create_driver = PythonOperator(
        task_id='create_driver_aggregates',
        python_callable=create_driver_aggregates,
        provide_context=True
    )
    
    create_hourly = PythonOperator(
        task_id='create_hourly_demand',
        python_callable=create_hourly_demand,
        provide_context=True
    )
    
    create_location = PythonOperator(
        task_id='create_location_metrics',
        python_callable=create_location_metrics,
        provide_context=True
    )
    
    # Validation
    validate = PythonOperator(
        task_id='validate_star_schema',
        python_callable=validate_star_schema,
        provide_context=True
    )
    
    # Optimization
    optimize = PythonOperator(
        task_id='optimize_gold_tables',
        python_callable=optimize_gold_tables,
        provide_context=True
    )
    
    # Completeness check
    check_complete = PythonOperator(
        task_id='check_gold_completeness',
        python_callable=check_gold_completeness,
        provide_context=True
    )
    
    end = DummyOperator(task_id='end')
    
    # Define dependencies
    wait_for_silver >> start >> create_dims
    create_dims >> [create_trips, create_payments]
    [create_trips, create_payments] >> [create_daily, create_driver, create_hourly, create_location]
    [create_daily, create_driver, create_hourly, create_location] >> validate
    validate >> optimize >> check_complete >> end