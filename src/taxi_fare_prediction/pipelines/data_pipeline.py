from pathlib import Path
from taxi_fare_prediction import logger
from taxi_fare_prediction.utils.custom_exceptions import CustomJSONError
from taxi_fare_prediction.core.constants import constants
from taxi_fare_prediction.core.config import ConfigurationManager
from taxi_fare_prediction.data.data_ingestion import DataIngestion
from taxi_fare_prediction.data.data_validation import DataValiadtion
from taxi_fare_prediction.data.data_preprocessing import DataPreProcessing
from taxi_fare_prediction.data.data_transformation import DataTransformation

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_load_config = config.get_data_ingestion_config()
        data_load = DataIngestion(config=data_load_config)
        data_load.download_file()
    

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()


class DataPreProcessTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreProcessing(config=data_preprocessing_config)
        


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path(constants.STATUS_FILE_PATH), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_spliting()

            else:
                raise CustomJSONError("500","You data schema is not valid")

        except CustomJSONError as e:
            logger.error(f"Custom JSON Error: {e.to_json()}")
            return e.to_json()
        
        except Exception as e:
            print(e)