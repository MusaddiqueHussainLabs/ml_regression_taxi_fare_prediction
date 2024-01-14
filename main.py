from taxi_fare_prediction import logger
from taxi_fare_prediction.pipelines.data_pipeline import DataIngestionTrainingPipeline, DataValidationTrainingPipeline, DataTransformationTrainingPipeline
from taxi_fare_prediction.pipelines.model_pipeline import ModelTrainerTrainingPipeline, ModelEvaluationTrainingPipeline
from taxi_fare_prediction.modeling.predict_model import PredictionPipeline

import pandas as pd

# STAGE_NAME = "Data Ingestion stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_load = DataIngestionTrainingPipeline()
#    data_load.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Data Validation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataValidationTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

# STAGE_NAME = "Data Transformation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataTransformationTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelTrainerTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


# STAGE_NAME = "Model evaluation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = ModelEvaluationTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# sample_data = {
#     'vendor_id': ['VTS'],
#     'rate_code': [1],
#     'passenger_count': [1],
#     'trip_time_in_secs': [1140],
#     'trip_distance': [3.75],
#     'payment_type': ['CRD']
# }
# sample_df = pd.DataFrame(sample_data)
# prediction_pipeline = PredictionPipeline()
# prediction_result = prediction_pipeline.predict(sample_df)
# print(prediction_result)