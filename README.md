# Distributed Mobility Data Pipeline
Production grade data pipeline for large scale mobility events using PySpark, Delta Lake, and a medallion architecture.

## Overview
This project implements a production style data engineering pipeline for large scale mobility and ride activity data. The system processes raw event data through a medallion architecture (Bronze, Silver, Gold), models it using a star schema and produces analytics ready aggregate tables.

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
```

## Project Structure
```bash
distributed-mobility-data-pipeline/
├── airflow/
│   └── dags/
│       └── bronze_ingestion_dag.py
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
│   └── test_gold_schema.py
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
