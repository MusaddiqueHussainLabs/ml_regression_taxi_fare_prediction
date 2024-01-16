import os
from box.exceptions import BoxValueError
import yaml
from taxi_fare_prediction import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def read_params_from_file(model_name):
    with open('src/taxi_fare_prediction/core/params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    return params.get(model_name, {})

@ensure_annotations
def get_user_friendly_model_name(model_name):

    if model_name == 'LinearRegression':
        user_friendly_model_name = 'linear_regression'
    elif model_name == 'Ridge':
        user_friendly_model_name = 'ridge'
    elif model_name == 'Lasso':
        user_friendly_model_name = 'lasso'
    elif model_name == 'RandomForestRegressor':
        user_friendly_model_name = 'random_forest_regressor'
    elif model_name == 'SGDRegressor':
        user_friendly_model_name = 'sgd_regressor'
    elif model_name == 'ElasticNet':
        user_friendly_model_name = 'elastic_net'
    elif model_name == 'GradientBoostingRegressor':
        user_friendly_model_name = 'gradient_boosting_regressor'
    elif model_name == 'SVR':
        user_friendly_model_name = 'svr'
    elif model_name == 'KNeighborsRegressor':
        user_friendly_model_name = 'k_neighbors_regressor'
    else:
        user_friendly_model_name = model_name

    return user_friendly_model_name