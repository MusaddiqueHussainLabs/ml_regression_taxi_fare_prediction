import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import json
from taxi_fare_prediction.core.config import ModelEvaluationConfig
from taxi_fare_prediction.utils.helper_functions import save_json
from taxi_fare_prediction.core.config import ConfigurationManager
from taxi_fare_prediction.data.data_preprocessing import DataPreProcessing
from pathlib import Path
from box import ConfigBox


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self, user_friendly_model_name, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)

        # Update eval.json
        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

        metric_file_path = Path(f"{self.config.root_dir}/eval_{user_friendly_model_name}.json")
        save_json(metric_file_path, metrics)

        return ConfigBox(metrics)

    
