import pandas as pd
import os
from taxi_fare_prediction import logger
from sklearn.linear_model import ElasticNet
import joblib
from taxi_fare_prediction.core.config import ConfigurationManager
from taxi_fare_prediction.schemas.config_schema import ModelTrainerConfig
from taxi_fare_prediction.data.data_preprocessing import DataPreProcessing
from taxi_fare_prediction.data.data_transformation import DataTransformation


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

        model = data_transformation.create_pipeline()
        model.fit(X_train, y_train)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

        return model, X_test, y_test

        

