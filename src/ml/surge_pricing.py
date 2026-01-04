from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
import mlflow
import mlflow.spark
from src.utils.spark_session import get_spark_session
from src.utils.delta_utils import DeltaLakeManager
from src.utils.logger import setup_logger
import yaml

logger = setup_logger(__name__, 'logs')

class SurgePricingModel:
    
    def __init__(self, config_path='config/config.yaml'):

        logger.info("Initializing Surge Pricing Model...")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        spark_cfg = self.config['spark']

        self.spark = get_spark_session(
            app_name="SurgePricingModel",
            master=spark_cfg['master'],
            shuffle_partitions=spark_cfg['shuffle_partitions'],
            timezone=spark_cfg['timezone']
        )

        self.delta_manager = DeltaLakeManager(self.spark)
        
        # MLflow configuration
        mlflow_cfg = self.config.get('mlflow')
        if mlflow_cfg:
            mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
            mlflow.set_experiment(mlflow_cfg['experiment_name'])
            self.use_mlflow = True
        else:
            logger.warning("MLflow config not found in config.yaml. Running without MLflow logging.")
            self.use_mlflow = False

        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        self.test_size = 0.2
        self.random_state = 42
    
    def load_features(self):

        logger.info("Loading surge pricing features...")

        df = self.delta_manager.read_delta('./data/features/surge_pricing')

        logger.info(f"Columns: {df.columns}")
        df.printSchema()

        logger.info(f"Loaded {df.count():,} feature records")
        return df
    
    def prepare_data_regression(self, df):
        
        logger.info("Preparing data for regression...")
        
        # Encode categorical variables
        pickup_indexer = StringIndexer(inputCol='pickup_location_id', outputCol='pickup_loc_idx')
        pickup_encoder = OneHotEncoder(inputCol='pickup_loc_idx', outputCol='pickup_loc_vec')
        
        # Feature columns
        numeric_features = [
            'hour', 'day_of_week', 'is_weekend',
            'trip_count', 'driver_count', 'demand_supply_ratio',
            'avg_fare', 'avg_distance', 'avg_duration',
            'is_peak_hour', 'is_late_night',
            'surge_lag_1', 'demand_supply_lag_1',
            'surge_avg_7d', 'demand_supply_avg_7d', 'trip_count_avg_7d',
            'likely_bad_weather'
        ]
        
        categorical_features = ['pickup_loc_vec']
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=numeric_features + categorical_features,
            outputCol='features'
        )
        
        # Create pipeline
        preprocessing_pipeline = Pipeline(stages=[
            pickup_indexer,
            pickup_encoder,
            assembler
        ])
        
        # Fit and transform
        preprocessor = preprocessing_pipeline.fit(df)
        df_processed = preprocessor.transform(df)
        
        # Select features and target
        df_final = df_processed.select(
            col('features'),
            col('avg_surge').alias('label'),
            col('trip_date'),
            col('pickup_location_id'),
            col('demand_supply_ratio')
        )
        
        logger.info("Data preparation complete (regression)")
        
        return df_final, preprocessor
    
    def prepare_data_classification(self, df):
        
        logger.info("Preparing data for classification...")
        
        # Surge categories
        df_with_category = df.withColumn(
            'surge_category',
            when(col('avg_surge') <= 1.0, 0)
            .when(col('avg_surge') <= 1.5, 1)
            .when(col('avg_surge') <= 2.0, 2)
            .otherwise(3)
        )
        
        # Same preprocessing as regression
        pickup_indexer = StringIndexer(inputCol='pickup_location_id', outputCol='pickup_loc_idx')
        pickup_encoder = OneHotEncoder(inputCol='pickup_loc_idx', outputCol='pickup_loc_vec')
        
        numeric_features = [
            'hour', 'day_of_week', 'is_weekend',
            'trip_count', 'driver_count', 'demand_supply_ratio',
            'avg_fare', 'avg_distance', 'avg_duration',
            'is_peak_hour', 'is_late_night',
            'surge_lag_1', 'demand_supply_lag_1',
            'surge_avg_7d', 'demand_supply_avg_7d', 'trip_count_avg_7d',
            'likely_bad_weather'
        ]
        
        categorical_features = ['pickup_loc_vec']
        
        assembler = VectorAssembler(
            inputCols=numeric_features + categorical_features,
            outputCol='features'
        )
        
        preprocessing_pipeline = Pipeline(stages=[
            pickup_indexer,
            pickup_encoder,
            assembler
        ])
        
        preprocessor = preprocessing_pipeline.fit(df_with_category)
        df_processed = preprocessor.transform(df_with_category)
        
        df_final = df_processed.select(
            col('features'),
            col('surge_category').alias('label'),
            col('avg_surge'),
            col('trip_date'),
            col('pickup_location_id')
        )
        
        logger.info("Data preparation complete (classification)")
        
        return df_final, preprocessor
    
    def split_data(self, df):
        
        logger.info(f"Splitting data (test_size={self.test_size})...")
        
        train_df, test_df = df.randomSplit(
            [1 - self.test_size, self.test_size],
            seed=self.random_state
        )
        
        logger.info(f"Train set: {train_df.count():,} records")
        logger.info(f"Test set: {test_df.count():,} records")
        
        return train_df, test_df
    
    def train_regression_model(self, train_df, test_df):
        logger.info("Training Surge Regression Model (Random Forest)")
        
        with mlflow.start_run(run_name="surge_regression_rf"):
            
            params = {
                'model_type': 'RandomForest_Regression',
                'num_trees': 100,
                'max_depth': 10,
                'task': 'regression'
            }
            
            mlflow.log_params(params)
            
            # Create model
            rf = RandomForestRegressor(
                featuresCol='features',
                labelCol='label',
                numTrees=params['num_trees'],
                maxDepth=params['max_depth'],
                seed=self.random_state
            )
            
            # Train
            logger.info("Training model...")
            model = rf.fit(train_df)
            
            # Predict
            predictions = model.transform(test_df)
            
            # Evaluate
            evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction')
            
            rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
            mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            
            # Log model
            mlflow.spark.log_model(model, "model")
            
            return model, predictions, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def train_classification_model(self, train_df, test_df):
        logger.info("Training Surge Classification Model (Random Forest)")
        
        with mlflow.start_run(run_name="surge_classification_rf"):
            
            params = {
                'model_type': 'RandomForest_Classification',
                'num_trees': 100,
                'max_depth': 10,
                'task': 'classification'
            }
            
            mlflow.log_params(params)
            
            # Create model
            rf = RandomForestClassifier(
                featuresCol='features',
                labelCol='label',
                numTrees=params['num_trees'],
                maxDepth=params['max_depth'],
                seed=self.random_state
            )
            
            # Train
            logger.info("Training model...")
            model = rf.fit(train_df)
            
            # Predict
            predictions = model.transform(test_df)
            
            # Evaluate
            evaluator = MulticlassClassificationEvaluator(
                labelCol='label',
                predictionCol='prediction'
            )
            
            accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
            f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
            precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
            recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            
            # Log model
            mlflow.spark.log_model(model, "model")
            
            return model, predictions, {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
    
    def train_all_models(self):
        logger.info("Starting Surge Pricing Model Training")
        
        try:
            # Load features
            df = self.load_features()
            
            # Regression approach
            logger.info("\n--- Regression Approach ---\n")
            df_reg, preprocessor_reg = self.prepare_data_regression(df)
            train_reg, test_reg = self.split_data(df_reg)
            reg_model, reg_pred, reg_metrics = self.train_regression_model(train_reg, test_reg)
            
            # Classification approach
            logger.info("\n--- Classification Approach ---\n")
            df_cls, preprocessor_cls = self.prepare_data_classification(df)
            train_cls, test_cls = self.split_data(df_cls)
            cls_model, cls_pred, cls_metrics = self.train_classification_model(train_cls, test_cls)
            
            # Summary
            logger.info("Surge Pricing Model Results")
            logger.info("\nRegression Model:")
            logger.info(f"  RMSE: {reg_metrics['rmse']:.4f}")
            logger.info(f"  MAE: {reg_metrics['mae']:.4f}")
            logger.info(f"  R²: {reg_metrics['r2']:.4f}")
            
            logger.info("\nClassification Model:")
            logger.info(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {cls_metrics['f1']:.4f}")
            
            return {
                'regression': reg_metrics,
                'classification': cls_metrics
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

def main():    
    model_trainer = SurgePricingModel()
    results = model_trainer.train_all_models()
    
    print("\nSurge Pricing Model Training Complete!")
    print(f"Regression Metrics: {results['regression']}")
    print(f"Classification Metrics: {results['classification']}")

if __name__ == "__main__":
    main()