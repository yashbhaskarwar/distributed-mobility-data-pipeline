from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.external_task import ExternalTaskSensor
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.feature_engineering import FeatureEngineer
from src.ml.demand_forecasting import DemandForecastingModel
from src.ml.surge_pricing import SurgePricingModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__,"logs")

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

def create_ml_features(**context):
    logger.info("Creating ML features...")
    
    engineer = FeatureEngineer()
    engineer.create_all_features()
    
    logger.info("ML features created")


def validate_features(**context):
    from src.utils.spark_session import get_spark_session
    from src.utils.delta_utils import DeltaLakeManager
    from src.utils.data_quality import DataQualityChecker
    
    logger.info("Validating ML features...")
    
    spark = get_spark_session()
    delta = DeltaLakeManager(spark)
    quality = DataQualityChecker(spark)
    
    # Check demand features
    demand_df = delta.read_delta('./data/features/demand_forecasting')
    demand_report = quality.generate_quality_report(
        demand_df,
        'demand_features',
        key_columns=['trip_date', 'city', 'zone_name', 'hour']
    )
    
    # Check surge features
    surge_df = delta.read_delta('./data/features/surge_pricing')
    surge_report = quality.generate_quality_report(
        surge_df,
        'surge_features',
        key_columns=['trip_date', 'city', 'zone_name', 'hour']
    )
    
    logger.info("Feature validation complete")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='demand_report', value=demand_report)
    context['task_instance'].xcom_push(key='surge_report', value=surge_report)


def train_demand_model(**context):
    logger.info("Training demand forecasting models...")
    
    model_trainer = DemandForecastingModel()
    results, best_model = model_trainer.train_all_models()
    
    logger.info(f"Best demand model: {best_model}")
    logger.info(f"Metrics: {results[best_model]}")
    
    # Push results to XCom
    context['task_instance'].xcom_push(key='best_model', value=best_model)
    context['task_instance'].xcom_push(key='metrics', value=results[best_model])


def train_surge_model(**context):
    logger.info("Training surge pricing models...")
    
    model_trainer = SurgePricingModel()
    results = model_trainer.train_all_models()
    
    logger.info(f"Surge models trained")
    logger.info(f"Regression metrics: {results['regression']}")
    logger.info(f"Classification metrics: {results['classification']}")
    
    # Push results to XCom
    context['task_instance'].xcom_push(key='regression_metrics', value=results['regression'])
    context['task_instance'].xcom_push(key='classification_metrics', value=results['classification'])


def register_best_models(**context):
    logger.info("Registering best models...")
    
    # Get best models from XCom
    ti = context['task_instance']
    demand_best = ti.xcom_pull(task_ids='train_demand_model', key='best_model')
    demand_metrics = ti.xcom_pull(task_ids='train_demand_model', key='metrics')
    
    logger.info(f"Best demand model: {demand_best}")
    logger.info(f"Metrics: {demand_metrics}")
    
    
    logger.info("Models registered (simulation)")


def generate_model_report(**context):
    logger.info("Generating model training report...")
    
    ti = context['task_instance']
    
    # Pull all metrics
    demand_best = ti.xcom_pull(task_ids='train_demand_model', key='best_model')
    demand_metrics = ti.xcom_pull(task_ids='train_demand_model', key='metrics')
    surge_reg = ti.xcom_pull(task_ids='train_surge_model', key='regression_metrics')
    surge_cls = ti.xcom_pull(task_ids='train_surge_model', key='classification_metrics')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'demand_forecasting': {
            'best_model': demand_best,
            'metrics': demand_metrics
        },
        'surge_pricing': {
            'regression': surge_reg,
            'classification': surge_cls
        }
    }

    logger.info("ML Training Report")
    logger.info(f"\nDemand Forecasting:")
    logger.info(f"  Best Model: {demand_best}")
    logger.info(f"  RMSE: {demand_metrics.get('rmse', 'N/A')}")
    logger.info(f"  R²: {demand_metrics.get('r2', 'N/A')}")
    logger.info(f"\nSurge Pricing:")
    logger.info(f"  Regression RMSE: {surge_reg.get('rmse', 'N/A')}")
    logger.info(f"  Classification Accuracy: {surge_cls.get('accuracy', 'N/A')}")
    
    # Save report
    import json
    with open(f"logs/ml_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Report generated and saved")

# Create DAG
with DAG(
    'ml_model_training',
    default_args=default_args,
    description='Train ML models for demand forecasting and surge pricing',
    schedule_interval='0 4 * * 0',  # Weekly on Sunday at 4 AM
    catchup=False,
    tags=['ml', 'training', 'weekly'],
) as dag:
    
    # Wait for Gold layer to be ready
    wait_for_gold = ExternalTaskSensor(
        task_id='wait_for_gold',
        external_dag_id='gold_layer_aggregation',
        external_task_id='end',
        timeout=600,
        mode='poke',
        poke_interval=60
    )
    
    start = DummyOperator(task_id='start')
    
    # Feature engineering
    create_features = PythonOperator(
        task_id='create_ml_features',
        python_callable=create_ml_features,
        provide_context=True
    )
    
    # Validate features
    validate = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        provide_context=True
    )
    
    # Train models
    train_demand = PythonOperator(
        task_id='train_demand_model',
        python_callable=train_demand_model,
        provide_context=True
    )
    
    train_surge = PythonOperator(
        task_id='train_surge_model',
        python_callable=train_surge_model,
        provide_context=True
    )
    
    # Register models
    register_models = PythonOperator(
        task_id='register_best_models',
        python_callable=register_best_models,
        provide_context=True
    )
    
    # Generate report
    generate_report = PythonOperator(
        task_id='generate_model_report',
        python_callable=generate_model_report,
        provide_context=True
    )
    
    end = DummyOperator(task_id='end')
    
    # Define dependencies
    wait_for_gold >> start >> create_features >> validate
    validate >> [train_demand, train_surge]
    [train_demand, train_surge] >> register_models >> generate_report