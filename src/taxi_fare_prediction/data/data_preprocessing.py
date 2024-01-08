import pandas as pd
from scipy.stats import zscore
from taxi_fare_prediction.schemas.config_schema import DataPreProcessingConfig


class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        self.config = config

    def load_data(self):
        data = pd.read_csv(self.config.input_file_path, encoding="utf-8")
        return data
    
    def handle_missing_values(self, data):
        return data.dropna()

    def remove_outliers(self, data, columns_to_check, threshold=3):
        
        # Assuming a simple z-score based outlier detection
        z_scores = zscore(data[columns_to_check])
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < threshold).all(axis=1)
        return data[filtered_entries]
    
    def split_X_y(self, data):
        
        X = data.drop(columns=self.config.target_column)
        y = data[self.config.target_column]
        return X, y