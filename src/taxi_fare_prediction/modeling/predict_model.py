import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('model_artifacts/trained_models/taxi_fare_prediction_model_v1.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction