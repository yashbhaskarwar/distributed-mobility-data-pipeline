from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.external_task import ExternalTaskSensor
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.transformation.bronze_to_silver import SilverLayerTransformer
from src.utils.data_quality import DataQualityChecker
from src.utils.spark_session import get_spark_session
from src.utils.delta_utils import DeltaLakeManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

default_args = {
    'owner': 'data-team',
    'depends_on_past': True,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def transform_dimension_tables(**context):
    logger.info("Transforming dimensional tables to Silver...")
    
    transformer = SilverLayerTransformer()
    
    # Transform dimensions
    transformer.transform_users()
    transformer.transform_drivers()
    transformer.transform_vehicles()
    transformer.transform_locations()
    
    logger.info("Dimensional tables transformed")


def transform_fact_tables(**context):
    logger.info("Transforming fact tables to Silver...")
    
    transformer = SilverLayerTransformer()
    
    # Transform facts
    transformer.transform_trips()
    transformer.transform_payments()
    
    logger.info("Fact tables transformed")

def validate_silver_users(**context):
    logger.info("Validating Silver users...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    quality = DataQualityChecker(spark)
    
    df = delta.read_delta('./data/silver_delta/users')
    report = quality.generate_quality_report(df, 'silver_users', ['user_id'])
    
    # Push report to XCom
    context['task_instance'].xcom_push(key='users_report', value=report)
    logger.info("Users validation complete")


def validate_silver_drivers(**context):
    logger.info("Validating Silver drivers...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    quality = DataQualityChecker(spark)
    
    df = delta.read_delta('./data/silver_delta/drivers')
    report = quality.generate_quality_report(df, 'silver_drivers', ['driver_id'])
    
    context['task_instance'].xcom_push(key='drivers_report', value=report)
    logger.info("Drivers validation complete")


def validate_silver_trips(**context):
    logger.info("Validating Silver trips...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    quality = DataQualityChecker(spark)
    
    df = delta.read_delta('./data/silver_delta/trips')
    
    # Comprehensive validation
    report = quality.generate_quality_report(df, 'silver_trips', ['trip_id'])
    
    # Additional checks
    quality.check_numeric_ranges(df, 'distance_km', min_value=0, max_value=200)
    quality.check_numeric_ranges(df, 'duration_minutes', min_value=0, max_value=300)
    quality.check_numeric_ranges(df, 'total_amount', min_value=0, max_value=1000)
    
    context['task_instance'].xcom_push(key='trips_report', value=report)
    logger.info("Trips validation complete")


def check_data_freshness(**context):
    from pyspark.sql.functions import max as spark_max
    
    logger.info("Checking data freshness...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    
    trips_df = delta.read_delta('./data/silver_delta/trips')
    
    # Get latest trip date
    latest_trip = trips_df.agg(spark_max('trip_date')).collect()[0][0]
    
    logger.info(f"Latest trip date: {latest_trip}")
    
    # Alert
    from datetime import date
    days_old = (date.today() - latest_trip).days
    
    if days_old > 7:
        logger.warning(f"Data is {days_old} days old!")
    else:
        logger.info(f"Data freshness OK ({days_old} days old)")
    
    context['task_instance'].xcom_push(key='days_old', value=days_old)


def generate_silver_summary(**context):
    logger.info("Generating Silver layer summary...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    
    tables = ['users', 'drivers', 'vehicles', 'locations', 'trips', 'payments']
    summary = {}
    
    for table in tables:
        df = delta.read_delta(f'./data/silver_delta/{table}')
        summary[table] = {
            'record_count': df.count(),
            'columns': len(df.columns)
        }
    
    logger.info(f"Silver Summary: {summary}")
    context['task_instance'].xcom_push(key='summary', value=summary)

# Create DAG
with DAG(
    'silver_layer_transformation',
    default_args=default_args,
    description='Transform Bronze data to Silver with quality checks',
    schedule_interval='0 2 * * *',  # Daily at 2 AM 
    catchup=False,
    tags=['silver', 'transformation', 'daily'],
) as dag:
    
    wait_for_bronze = ExternalTaskSensor(
        task_id='wait_for_bronze',
        external_dag_id='bronze_layer_ingestion',
        external_task_id='end',
        timeout=600,
        mode='poke',
        poke_interval=60
    )
    
    start = DummyOperator(task_id='start')
    
    # Transformation tasks
    transform_dims = PythonOperator(
        task_id='transform_dimension_tables',
        python_callable=transform_dimension_tables,
        provide_context=True
    )
    
    transform_facts = PythonOperator(
        task_id='transform_fact_tables',
        python_callable=transform_fact_tables,
        provide_context=True
    )
    
    # Validation tasks (parallel)
    validate_users = PythonOperator(
        task_id='validate_silver_users',
        python_callable=validate_silver_users,
        provide_context=True
    )
    
    validate_drivers = PythonOperator(
        task_id='validate_silver_drivers',
        python_callable=validate_silver_drivers,
        provide_context=True
    )
    
    validate_trips = PythonOperator(
        task_id='validate_silver_trips',
        python_callable=validate_silver_trips,
        provide_context=True
    )
    
    # Data quality checks
    freshness_check = PythonOperator(
        task_id='check_data_freshness',
        python_callable=check_data_freshness,
        provide_context=True
    )
    
    # Summary
    summary = PythonOperator(
        task_id='generate_silver_summary',
        python_callable=generate_silver_summary,
        provide_context=True
    )
    
    end = DummyOperator(task_id='end')
    
    # Define dependencies
    wait_for_bronze >> start
    start >> transform_dims >> transform_facts
    transform_facts >> [validate_users, validate_drivers, validate_trips]
    [validate_users, validate_drivers, validate_trips] >> freshness_check
    freshness_check >> summary >> end