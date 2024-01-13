import os
from taxi_fare_prediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from taxi_fare_prediction.schemas.config_schema import DataTransformationConfig

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    # def train_test_spliting(self):
    #     data = pd.read_csv(self.config.input_file_path)

    #     # Split the data into training and test sets. (0.75, 0.25) split.
    #     train, test = train_test_split(data)

    #     train.to_csv(os.path.join(self.config.root_dir, "taxi-fare-train.csv"),index = False)
    #     test.to_csv(os.path.join(self.config.root_dir, "taxi-fare-test.csv"),index = False)



    #     logger.info("Splited data into training and test sets")
    #     logger.info(train.shape)
    #     logger.info(test.shape)

    #     print(train.shape)
    #     print(test.shape)
    
    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train.to_csv(os.path.join(self.config.root_dir, "taxi-fare-train.csv"),index = False)
        X_test.to_csv(os.path.join(self.config.root_dir, "taxi-fare-test.csv"),index = False)

        logger.info("Splited data into training and test sets")

        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self):
        numeric_features = ['passenger_count', 'trip_time_in_secs', 'trip_distance']
        categorical_features = ['vendor_id', 'rate_code', 'payment_type']

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('Random Forest', RandomForestRegressor())
        ])
        return model