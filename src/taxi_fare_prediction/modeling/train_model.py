import pandas as pd
import os
import joblib
from joblib import dump
import json
from pathlib import Path

from taxi_fare_prediction import logger
from taxi_fare_prediction.core.config import ConfigurationManager
from taxi_fare_prediction.schemas.config_schema import ModelTrainerConfig
from taxi_fare_prediction.data.data_preprocessing import DataPreProcessing
from taxi_fare_prediction.data.data_transformation import DataTransformation
from taxi_fare_prediction.modeling.evaluate_model import ModelEvaluation
from taxi_fare_prediction.experiments.experiment_tracking import ExperimentManager
from taxi_fare_prediction.utils.helper_functions import save_json, get_user_friendly_model_name


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):

        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreProcessing(config=data_preprocessing_config)

        data = data_preprocessing.load_data()

        columns_to_check = [self.config.target_column]
        data = data_preprocessing.remove_outliers(data, columns_to_check)

        X, y = data_preprocessing.split_X_y(data)

        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)

        X_train, X_test, y_train, y_test = data_transformation.split_train_test(X, y, test_size=0.2, random_state=42)

        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        experiment_manager_config = config.get_experiment_manager_config()
        experiment_manager = ExperimentManager(config=experiment_manager_config)

        experiment_manager.create_experiment()
        
        numeric_features = [self.config.all_schema.NUMERIC_FEATURES.passenger_count, 
                            self.config.all_schema.NUMERIC_FEATURES.trip_time_in_secs, 
                            self.config.all_schema.NUMERIC_FEATURES.trip_distance]
        
        categorical_features = [self.config.all_schema.CATEGORICAL_FEATURES.vendor_id, 
                                self.config.all_schema.CATEGORICAL_FEATURES.rate_code, 
                                self.config.all_schema.CATEGORICAL_FEATURES.payment_type]

        preprocessor = data_transformation.get_preprocessor(numeric_features, categorical_features)
        model_pipelines = data_transformation.create_model_pipelines(preprocessor)

        # Initialize variables to store best model and metrics
        best_model = None
        best_metrics = {"mae": float('inf'), "mse": float('inf'), "rmse": float('inf'), "r2": float('-inf')}

        for model_name, model_pipeline in model_pipelines.items():

            # uncomment the below line, if different params needs to pass other than default params and comment get_model_name_and_params(model_name) 
            # params = self.config.all_models_params.get(model_name)
            # model_pipeline.set_params(**params)

            model, params = data_transformation.get_model_name_and_params(model_name)

            user_friendly_model_name = get_user_friendly_model_name(model_name)

            params_path = Path(f"{self.config.eval_root_dir}/best_param_{user_friendly_model_name}.json")
            save_json(params_path, params)

            model_pipeline.fit(X_train, y_train)

            y_pred = model_pipeline.predict(X_test)

            metrics = model_evaluation.eval_metrics(user_friendly_model_name, y_test, y_pred)

            # MLflow Tracking to keep track of training
            experiment_manager.run_experiment(model_name=user_friendly_model_name, X_test=X_test, y_pred=y_pred, model=model_pipeline, params=params, metrics=metrics)

            # Check if this model is the best so far
            if metrics.mae < best_metrics["mae"] and metrics.mse < best_metrics["mse"] and metrics.r2 > best_metrics["r2"]:
                best_metrics = metrics
                best_model = model_pipeline

        #not saving in project dir, will modify to track with DVC
        # dump(best_model, os.path.join(self.config.root_dir, f"taxi_fare_{user_friendly_model_name}.joblib"))                    

        return best_model, X_test, y_test

        

