import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from taxi_fare_prediction.core.config import ModelEvaluationConfig
from taxi_fare_prediction.utils.helper_functions import save_json
from taxi_fare_prediction.core.config import ConfigurationManager
from taxi_fare_prediction.data.data_preprocessing import DataPreProcessing
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    

    def log_into_mlflow(self):

        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreProcessing(config=data_preprocessing_config)

        data = data_preprocessing.load_data()

        columns_to_check = [self.config.target_column]
        data = data_preprocessing.remove_outliers(data, columns_to_check)

        X, y = data_preprocessing.split_X_y(data)
        
        model = joblib.load(self.config.model_path)
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(X)

            (rmse, mae, r2) = self.eval_metrics(y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="SGDRegressorModel")
            else:
                mlflow.sklearn.log_model(model, "model")

    
