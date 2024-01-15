import pandas as pd
import os
import joblib
from joblib import dump
import json

from taxi_fare_prediction import logger
from taxi_fare_prediction.core.config import ConfigurationManager
from taxi_fare_prediction.schemas.config_schema import ModelTrainerConfig
from taxi_fare_prediction.data.data_preprocessing import DataPreProcessing
from taxi_fare_prediction.data.data_transformation import DataTransformation
from taxi_fare_prediction.modeling.evaluate_model import ModelEvaluation
from taxi_fare_prediction.experiments.experiment_tracking import ExperimentManager
from taxi_fare_prediction.utils.helper_functions import read_params_from_file


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV


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
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)

        experiment_manager_config = config.get_experiment_manager_config()
        experiment_manager = ExperimentManager(config=experiment_manager_config)

        taxi_fare_experiment = experiment_manager.create_experiment()

        numeric_features = ['passenger_count', 'trip_time_in_secs', 'trip_distance']
        categorical_features = ['vendor_id', 'rate_code', 'payment_type']

        preprocessor = data_transformation.get_preprocessor(numeric_features, categorical_features)

        # Initialize variables to store best model and metrics
        best_model = None
        best_metrics = {"mae": float('inf'), "mse": float('inf'), "rmse": float('inf'), "r2": float('-inf')}

        for model_name, params in self.config.all_models_params.models.items():
            
            # Create the model
            model = globals()[model_name]()
            # model.set_params(**params)

            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])

            model_pipeline.set_params(**params)

            best_model, best_params = self.train_model_with_params(model_pipeline, model_name, params, X_train, y_train)

            # Predict on the test set
            predictions = best_model.predict(X)

            metrics = model_evaluation_config.eval_metrics(model_name, y, predictions)

            # MLflow Tracking to keep track of training
            experiment_manager.run_experiment(model_name=model_name, X_test=X_test, model=best_model, params=best_params, metrics=metrics)

             # Check if this model is the best so far
            if metrics.mae < best_metrics["mae"] and metrics.mse < best_metrics["mse"] and metrics.r2 > best_metrics["r2"]:
                best_metrics = metrics
                best_model = model_name

        # Save the best model using joblib
        joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))

        return best_model, X_test, y_test
    
    def train_model_with_params(self, model_pipeline, model_name, params, X_train, y_train):
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(model_pipeline, param_grid=params, cv=2, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

         # Get best parameters
        best_params = grid_search.best_params_

        # Update best_param.json
        with open(f"{self.config.eval_root_dir}/best_param_{model_name}.json", "w") as json_file:
            json.dump(best_params, json_file)

        # Fit the model with the best parameters
        best_model = grid_search.best_estimator_

        return best_model, best_params

        

