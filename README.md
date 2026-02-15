# Distributed Mobility Data Pipeline
Production grade data pipeline for large scale mobility events using PySpark, Delta Lake and a medallion architecture.

## Overview
This project implements a production style data engineering pipeline for large scale mobility and ride activity data. The system processes raw event data through a medallion architecture (Bronze, Silver, Gold), models it using a star schema and produces analytics ready aggregate tables. The pipeline also includes feature engineering and machine learning workflows for demand forecasting and surge pricing.

##  System Architechture
```bash
Raw CSV Data
   |
   v
Bronze Layer (Delta Lake)
- Raw ingestion
- Audit columns
- Partitioned by date
   |
   v
Silver Layer (Delta Lake)
- Cleaned and validated data
- Deduplicated records
- Standardized formats
   |
   v
Gold Layer (Delta Lake)
- Star schema (dimensions + facts)
- Analytics aggregates
   |
   v
ML Layer
- Feature engineering
- Demand forecasting
- Surge pricing models
   |
   v
Batch Scoring (Spark + Delta)
- Large-scale inference
- Partitioned predictions
- Idempotent merge writes
   |
   v
Real-Time API 
- Demand prediction endpoint
- Surge prediction endpoint
- Batch inference endpoint
- Health checks and model metadata
   |
   v
Monitoring & Drift Detection
- MAE / RMSE / MAPE tracking
- PSI-based feature drift detection
- Missing rate shift monitoring
- Prediction volume monitoring
- API latency tracking
```

## Project Structure
```bash
distributed-mobility-data-pipeline/
├── airflow/
│   └── dags/
│       ├── bronze_ingestion_dag.py
│       ├── silver_dag.py
│       ├── ml_training_dag.py
│       └── gold_dag.py
├── config/
│   └── config.yaml
├── src/
│   ├── data_generation/
│   │   └── generate_all.py
│   ├── ingestion/
│   │   └── bronze_loader.py
│   ├── transformation/
│   │   ├── bronze_to_silver.py
│   │   ├── silver_to_gold.py
│   │   └── gold_aggregates.py
│   ├── ml/
│   │   ├── feature_engineering.py
│   │   ├── demand_forecasting.py
│   │   ├── batch_scoring.py
│   │   ├── model_monitoring.py
│   │   └── surge_pricing.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── spark_session.py
│       ├── delta_utils.py
│       └── data_quality.py
├── tests/
│   ├── test_data_generation.py
│   ├── test_bronze_ingestion.py
│   ├── test_silver_transformation.py
│   ├── test_gold_aggregates.py
│   ├── test_gold_schema.py
│   └── test_complete_pipeline.py
├── requirements.txt
└── README.md
```

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic mobility data
```bash
py -3.9 -m src.data_generation.generate_all
py -3.9 -m tests.test_data_generation
```

### 3. Run Bronze ingestion
```bash
py -3.9 -m src.ingestion.bronze_loader
py -3.9 -m tests.test_bronze_ingestion
```

### 4. Run Silver transformation
```bash
py -3.9 -m src.transformation.bronze_to_silver
py -3.9 -m tests.test_silver_transformation
```

### 5. Build Gold star schema
```bash
py -3.9 -m src.transformation.silver_to_gold
py -3.9 -m tests.test_gold_schema
```

### 6. Build Gold aggregate tables
```bash
py -3.9 -m src.transformation.gold_aggregates
py -3.9 -m tests.test_gold_aggregates
```

### 7. Run ML workflows locally
```bash
py -3.9 -m src.ml.feature_engineering
py -3.9 -m src.ml.demand_forecasting
py -3.9 -m src.ml.surge_pricing
```

### 8. Airflow Orchestration
```bash
1. Start the Airflow scheduler and webserver.
2. Enable the required DAGs.
3. Trigger the Bronze, Silver, ML training and Gold DAGs via schedule.
```

### 9. Start Real-Time Inference API
```bash
uvicorn src.api.app:app --reload
```
Health check
```bash
curl http://localhost:8000/health
```

### 10. Run batch scoring
```bash
py -3.9 -m src.ml.batch_scoring \
  --silver_features ./data/features/silver_features \
  --predictions_out ./data/gold/batch_predictions \
  --metrics_out ./data/gold/batch_scoring_metrics
```
Outputs:
- data/gold/batch_predictions (Delta)
- data/gold/batch_scoring_metrics (Delta)

### 11. Run Model Monitoring
```bash
py -3.9 -m src.ml.model_monitoring \
  --predictions_delta ./data/gold/batch_predictions \
  --monitoring_out ./data/gold/model_monitoring_metrics \
  --drift_out ./data/gold/model_drift_details
```

### 12. Run Complete Pipeline Tests
```bash
pytest tests/test_complete_pipeline.py
```
