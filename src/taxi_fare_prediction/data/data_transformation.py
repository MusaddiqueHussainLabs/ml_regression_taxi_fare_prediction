import os
from taxi_fare_prediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from taxi_fare_prediction.schemas.config_schema import DataTransformationConfig

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train.to_csv(os.path.join(self.config.root_dir, "taxi-fare-train.csv"),index = False)
        X_test.to_csv(os.path.join(self.config.root_dir, "taxi-fare-test.csv"),index = False)

        logger.info("Splited data into training and test sets")

        return X_train, X_test, y_train, y_test
    
    def create_model_pipelines(self, preprocessor):

        linear_reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        ridge_reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge())
        ])

        lasso_reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Lasso())
        ])

        sgd_reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SGDRegressor())
        ])

        elastic_net_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet())
        ])

        random_forest_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor())
        ])

        gradient_boosting_reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor())
        ])

        svr_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR())
        ])

        knn_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor())
        ])

        model_pipelines = {
            'LinearRegression': linear_reg_pipeline,
            'Ridge': ridge_reg_pipeline,
            'Lasso': lasso_reg_pipeline,
            'SGDRegressor': sgd_reg_pipeline,
            'ElasticNet': elastic_net_pipeline,
            'RandomForestRegressor': random_forest_pipeline,
            'GradientBoostingRegressor': gradient_boosting_reg_pipeline,
            'SVR': svr_pipeline,
            'KNeighborsRegressor': knn_pipeline
        }

        return model_pipelines

    def get_preprocessor(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    
    def get_model_name_and_params(self, model_name):

        model = globals()[model_name]()
        params = model.get_params()

        return model, params