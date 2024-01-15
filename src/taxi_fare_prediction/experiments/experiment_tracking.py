import mlflow
from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from taxi_fare_prediction.schemas.config_schema import ExperimentManagerConfig

class ExperimentManager:

    def __init__(self, config: ExperimentManagerConfig):
        self.config = config

    def get_mlflow_client(self):
        client = MlflowClient(tracking_uri=self.config.mlflow_uri)
        return client
    
    def create_experiment(self):

        client = self.get_mlflow_client()

        all_experiments = client.search_experiments()

        for experiment in all_experiments:
            if experiment.name == "taxi_fare_prediction":
                return experiment

        # Provide an Experiment description that will appear in the UI
        experiment_description = (
            "Taxi fare prediction experiment"
        )

        # Provide searchable tags that define characteristics of the Runs that
        # will be in this Experiment
        experiment_tags = {
            "project": "taxi_fare_prediction",
            "store_dept": "Regression",
            "team": "regression-ml",
            "data_source": "NYC Taxi Dataset",
            "author": "MusaddiqueHussain Jahagirdar",
            "mlflow.note.content": experiment_description,
        }

        # Create the Experiment, providing a unique name
        taxi_fare_experiment = client.create_experiment(
            name="taxi_fare_prediction", tags=experiment_tags
        )

        return taxi_fare_experiment
    
    def run_experiment(self, model_name, X_test, model, params, metrics):

        # Sets the current active experiment to the "taxi_fare_prediction" experiment and
        # returns the Experiment metadata
        taxi_fare_experiment = mlflow.set_experiment("taxi_fare_prediction")

        # Define a run name for this iteration of training.
        # If this is not set, a unique name will be auto-generated for your run.
        run_name = str(model_name).lower() + '_baseline'

        # Define an artifact path that the model will be saved to.
        artifact_path = run_name

        # Initiate the MLflow run context
        with mlflow.start_run(run_name=run_name) as run:
            # Log the parameters used for the model fit
            mlflow.log_params(params)

            # Log the error metrics that were calculated during validation
            mlflow.log_metrics(metrics)

            # Log an instance of the trained model for later use
            mlflow.sklearn.log_model(
                sk_model=model, input_example=X_test, artifact_path=artifact_path
            )
