# Taxi Fare Prediction with MLflow and DVC Integration through DagsHub

This repository contains code for building a machine learning model to predict taxi fares, integrated with MLflow and DVC through DagsHub. The project aims to provide a modular and scalable structure for regression model development, with enhanced experiment tracking and data versioning capabilities.

Utilizing machine learning algorithms to predict taxi fares based on relevant factors like distance, time, passenger, payment type, and historical data.

## Overview

The goal of this project is to develop an accurate taxi fare prediction model. The integration of MLflow and DVC, facilitated by DagsHub, offers improved experiment tracking, model versioning, and data versioning features.

## Features

- **Modular Project Structure:** The project follows a modular structure for organized development and easy maintenance.
- **MLflow Integration:** MLflow is used for experiment tracking, allowing users to log and compare multiple model runs.
- **DVC Integration:** DVC is employed for versioning and managing the dataset, providing reproducibility and collaboration benefits.
- **DagsHub Integration:** DagsHub is used as a platform for collaborative development, providing a centralized location for version-controlled code, data, and experiments.

## Branches

### 1. Main

The `main` branch is the default branch and provides the basic implementation of the taxi fare prediction model. It includes the essential code for data preprocessing, model training, and evaluation.

### 2. Grid Search LR RF

The `grid_search_lr_rf` branch explores hyperparameter tuning using grid search for both linear regression (`LR`) and random forest (`RF`) models. This branch aims to find the best hyperparameters for improved model performance.

### 3. Unified Model Baseline

The `unified_model_baseline` branch focuses on creating a unified model that incorporates the strengths of multiple models. It aims to achieve a balanced and robust model by combining the strengths of multiple models.

## Getting Started

Follow these steps to set up and run the project:

### Prerequisites

- Python (version 3.9 or higher)
- Pipenv (for virtual environment management)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MusaddiqueHussainLabs/ml_regression_taxi_fare_prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ml_regression_taxi_fare_prediction
   ```

3. Install dependencies using Pipenv:

   ```bash
   pip install -r requirements.txt
   ```

### MLflow and DVC Integration through DagsHub

#### MLflow

- Start MLflow server:

  ```bash
  mlflow ui
  ```

- localhost: Access MLflow UI at http://localhost:<port> to monitor and manage experiments.
- DagsHub: Access MLflow UI at https://dagshub.com/MusaddiqueHussainLabs/ml_regression_taxi_fare_prediction.mlflow to monitor and manage experiments.

#### DVC

- Initialize DVC:

  ```bash
  dvc init
  ```

- Add and track data files using DVC:

  ```bash
  dvc add data/raw/taxi_data.csv
  ```

- Commit changes:

  ```bash
  git add .
  git commit -m "Add data file using DVC"
  ```

### Running the Model

1. Choose the branch you want to work with:

   ```bash
   git checkout <branch_name>
   ```

   Replace `<branch_name>` with `main`, `grid_search_lr_rf`, or `unified_model_baseline`.

2. Run the model training script:

   ```bash
   python src/ml_regression_taxi_fare_prediction/modeling/train_model.py
   ```
   or

   uncomment the specific black in main.py and run below command

   ```bash
   python main.py
   ```

3. View MLflow UI to track experiments:

   - Visit https://dagshub.com/MusaddiqueHussainLabs/ml_regression_taxi_fare_prediction.mlflow (if not already running)
   - Explore experiments, metrics, and artifacts on DagsHub.

## Model Selection

Multiple models were tried during the development process, and the following models were considered:

- Linear Regression
- Ridge Regression
- Lasso Regression
- SGD Regressor
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor

The **Random Forest Regressor** demonstrated superior performance with good accuracy, and it was selected as the final model for taxi fare prediction.
Please refer a screen shots in **references** folder.

## Reports

The project generates reports, including evaluation metrics and figures. These can be found in the `reports/evaluation_metrics` and `reports/figures` directories, respectively.

## Note

**Still Working on DVC for Better Data and Model Tracking:**
While the current integration with DVC provides basic versioning, ongoing efforts are being made to enhance data and model tracking capabilities for improved reproducibility and collaboration. Stay tuned for updates! Contributions and feedback are welcome!

## Conclusion

This project provides a flexible framework for developing and experimenting with taxi fare prediction models. The integration with DagsHub enhances collaboration by centralizing code, data, and experiments. Contributions and feedback are welcome!