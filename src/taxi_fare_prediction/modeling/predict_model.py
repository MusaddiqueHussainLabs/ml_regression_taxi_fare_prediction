import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from taxi_fare_prediction import logger



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('models/taxi_fare_prediction_model_v1.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)
        logger.info(prediction)

        return prediction