from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
from src.utils.spark_session import get_spark_session
from src.utils.delta_utils import DeltaLakeManager
from src.utils.logger import setup_logger
import yaml

logger = setup_logger(__name__, 'logs')

class DemandForecastingModel:
    def __init__(self, config_path='config/config.yaml'): 
        logger.info("Initializing Demand Forecasting Model...")      
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.spark = get_spark_session(
            config_path,
            "local[*]",   
            200,          
            "UTC"         
        )
        self.delta_manager = DeltaLakeManager(self.spark)
        
        # MLflow configuration 
        mlflow_cfg = self.config.get("mlflow", {}) if isinstance(self.config, dict) else {}

        tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = mlflow_cfg.get("experiment_name", "demand_forecasting")
        mlflow.set_experiment(experiment_name)

        # ML config 
        ml_cfg = (self.config.get("ml", {}) if isinstance(self.config, dict) else {})
        df_cfg = ml_cfg.get("demand_forecasting", {})

        self.test_size = df_cfg.get("test_size", 0.2)
        self.random_state = df_cfg.get("random_state", 42)
        self.horizon_days = df_cfg.get("horizon_days", 7)

    def load_features(self):
        logger.info("Loading demand forecasting features...")
        
        df = self.delta_manager.read_delta('./data/features/demand_forecasting')
        logger.info(f"Columns: {df.columns}")
        logger.info(f"Loaded {df.count():,} feature records")
        
        return df
    
    def prepare_data(self, df):
        logger.info("Preparing data for modeling...")
        df = df.withColumn("pickup_location_id", col("pickup_location_id").cast("string"))

        # Feature columns 
        numeric_features = [
            "hour",
            "demand_lag_1", "demand_lag_7", "demand_lag_14",
            "demand_avg_7d", "demand_avg_30d", "demand_std_7d",
            "is_peak_hour", "is_night", "is_business_hours",
            "avg_surge", "avg_surge_lag_1",
        ]
        categorical_features = ["pickup_location_id"]  

        stages = []
        cat_feature_cols = []

        if categorical_features:
            indexers = [
                StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in categorical_features
            ]
            encoders = [
                OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_ohe"])
                for c in categorical_features
            ]
            stages.extend(indexers)
            stages.extend(encoders)
            cat_feature_cols = [f"{c}_ohe" for c in categorical_features]

        assembler = VectorAssembler(
            inputCols=numeric_features + cat_feature_cols,
            outputCol="features"
        )
        stages.append(assembler)

        preprocessing_pipeline = Pipeline(stages=stages)

        preprocessor = preprocessing_pipeline.fit(df)
        df_processed = preprocessor.transform(df)

        # Select features + label 
        df_final = df_processed.select(
            col("features"),
            col("trip_count").alias("label"),
            col("trip_date")
        )

        logger.info("Data preparation complete")
        return df_final, preprocessor
        
    def split_data(self, df):
        logger.info(f"Splitting data (test_size={self.test_size})...")
        
        # Time-based split (use last 20% of dates for testing)
        train_df, test_df = df.randomSplit(
            [1 - self.test_size, self.test_size],
            seed=self.random_state
        )
        
        logger.info(f"Train set: {train_df.count():,} records")
        logger.info(f"Test set: {test_df.count():,} records")
        
        return train_df, test_df
    
    def train_random_forest(self, train_df, test_df):
        logger.info("Training Random Forest Model")
        
        with mlflow.start_run(run_name="demand_rf"):
            
            # Log parameters
            params = {
                'model_type': 'RandomForest',
                'num_trees': 100,
                'max_depth': 10,
                'min_instances_per_node': 5
            }
            
            mlflow.log_params(params)
            
            # Create model
            rf = RandomForestRegressor(
                featuresCol='features',
                labelCol='label',
                numTrees=params['num_trees'],
                maxDepth=params['max_depth'],
                minInstancesPerNode=params['min_instances_per_node'],
                seed=self.random_state
            )
            
            # Train
            logger.info("Training Random Forest...")
            model = rf.fit(train_df)
            
            # Predict
            predictions = model.transform(test_df)
            
            # Evaluate
            evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction')
            
            rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
            mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            
            # Log model
            mlflow.spark.log_model(model, "model")
            
            # Feature importance
            feature_importance = model.featureImportances
            logger.info(f"Feature importance: {feature_importance}")
            
            return model, predictions, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def train_gradient_boosting(self, train_df, test_df):        
        logger.info("Training Gradient Boosting Model")
        
        with mlflow.start_run(run_name="demand_gbt"):
            
            # Log parameters
            params = {
                'model_type': 'GradientBoosting',
                'max_iter': 100,
                'max_depth': 5,
                'step_size': 0.1
            }
            
            mlflow.log_params(params)
            
            # Create model
            gbt = GBTRegressor(
                featuresCol='features',
                labelCol='label',
                maxIter=params['max_iter'],
                maxDepth=params['max_depth'],
                stepSize=params['step_size'],
                seed=self.random_state
            )
            
            # Train
            logger.info("Training Gradient Boosting...")
            model = gbt.fit(train_df)
            
            # Predict
            predictions = model.transform(test_df)
            
            # Evaluate
            evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction')
            
            rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
            mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            
            # Log model
            mlflow.spark.log_model(model, "model")
            
            return model, predictions, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def train_linear_regression(self, train_df, test_df):
        logger.info("Training Linear Regression (Baseline)")
        
        with mlflow.start_run(run_name="demand_lr"):
            
            # Log parameters
            params = {
                'model_type': 'LinearRegression',
                'max_iter': 100,
                'reg_param': 0.01
            }
            
            mlflow.log_params(params)
            
            # Create model
            lr = LinearRegression(
                featuresCol='features',
                labelCol='label',
                maxIter=params['max_iter'],
                regParam=params['reg_param']
            )
            
            # Train
            logger.info("Training Linear Regression...")
            model = lr.fit(train_df)
            
            # Predict
            predictions = model.transform(test_df)
            
            # Evaluate
            evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction')
            
            rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
            mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            
            # Log model
            mlflow.spark.log_model(model, "model")
            
            return model, predictions, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def train_all_models(self):
        logger.info("Starting Demand Forecasting Model Training")
        
        try:
            # Load and prepare data
            df = self.load_features()
            df_prepared, preprocessor = self.prepare_data(df)
            train_df, test_df = self.split_data(df_prepared)
            
            # Train multiple models
            results = {}
            
            # Random Forest
            rf_model, rf_pred, rf_metrics = self.train_random_forest(train_df, test_df)
            results['RandomForest'] = rf_metrics
            
            # Gradient Boosting
            gbt_model, gbt_pred, gbt_metrics = self.train_gradient_boosting(train_df, test_df)
            results['GradientBoosting'] = gbt_metrics
            
            # Linear Regression 
            lr_model, lr_pred, lr_metrics = self.train_linear_regression(train_df, test_df)
            results['LinearRegression'] = lr_metrics
            
            # Compare results
            logger.info("Model Comparison Results")
            
            for model_name, metrics in results.items():
                logger.info(f"\n{model_name}:")
                logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  R²: {metrics['r2']:.4f}")
            
            # Select best model
            best_model_name = min(results, key=lambda x: results[x]['rmse'])
            logger.info(f"\nBest Model: {best_model_name}")
            
            return results, best_model_name
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

def main():    
    model_trainer = DemandForecastingModel()
    results, best_model = model_trainer.train_all_models()
    
    print(f"\nBest performing model: {best_model}")
    print(f"Metrics: {results[best_model]}")

if __name__ == "__main__":
    main()
    