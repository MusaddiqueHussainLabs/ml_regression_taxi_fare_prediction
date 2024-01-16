import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from taxi_fare_prediction import logger

import mlflow


class PredictionPipeline:
    def __init__(self):
        # self.model = joblib.load(Path('models/taxi_fare_prediction_model_v1.joblib'))
        self.model = mlflow.sklearn.load_model("runs:/2ee58c73b1d84c62a11c844750ff73a2/random_forest_regressor_baseline")

    
    def predict(self, data):
        prediction = self.model.predict(data)
        logger.info(prediction)

        return prediction