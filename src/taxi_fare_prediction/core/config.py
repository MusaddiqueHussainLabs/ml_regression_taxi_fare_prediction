from taxi_fare_prediction.core.constants import constants
from taxi_fare_prediction.utils.helper_functions import read_yaml, create_directories
from taxi_fare_prediction.schemas.config_schema import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, DataPreProcessingConfig, ModelTrainerConfig, ModelEvaluationConfig, ExperimentManagerConfig

class ConfigurationManager:
    def __init__(self,
                 config_filepath = constants.CONFIG_FILE_PATH,
                 params_filepath = constants.PARAMS_FILE_PATH,
                 schema_filepath = constants.SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.data_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_load_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file
        )

        return data_load_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file_dir=config.status_file_dir,
            input_data_dir = config.input_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_preprocessing_config(self) -> DataPreProcessingConfig:
        config = self.config.data_preprocessing
        schema = self.schema.TARGET_COLUMN
        
        create_directories([config.root_dir])

        data_preprocessing_config = DataPreProcessingConfig(
            root_dir=config.root_dir,
            input_file_path = config.input_file_path,
            target_column = schema.name
        )

        return data_preprocessing_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            input_file_path=config.input_file_path,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        schema =  self.schema

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            eval_root_dir = config.eval_root_dir,
            all_models_params = params,
            all_schema = schema,
            target_column = schema.TARGET_COLUMN.name
            
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri="https://dagshub.com/MusaddiqueHussainLabs/ml_regression_taxi_fare_prediction.mlflow",
           
        )

        return model_evaluation_config
    
    def get_experiment_manager_config(self) -> ExperimentManagerConfig:
        params = self.params

        experiment_manager_config = ExperimentManagerConfig(
            
            mlflow_uri="https://dagshub.com/MusaddiqueHussainLabs/ml_regression_taxi_fare_prediction.mlflow",
            all_params=params
        )

        return experiment_manager_config
