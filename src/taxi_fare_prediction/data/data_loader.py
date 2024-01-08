import pandas as pd
from taxi_fare_prediction.schemas.config_schema import DataLoaderConfig


class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config

    def load_data(self):
        data = pd.read_csv(self.config.input_file_path, encoding="utf-8")
        return data