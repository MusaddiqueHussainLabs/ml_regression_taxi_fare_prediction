from pathlib import Path
from pydantic_settings import BaseSettings

class AppConstants(BaseSettings):

    CONFIG_FILE_PATH: Path =  Path("src/taxi_fare_prediction/core/config.yaml")
    PARAMS_FILE_PATH: Path =  Path("params.yaml")
    SCHEMA_FILE_PATH: Path =  Path("schema.yaml")
    STATUS_FILE_PATH: Path =  Path("data/interim/status.txt")

constants =    AppConstants()

