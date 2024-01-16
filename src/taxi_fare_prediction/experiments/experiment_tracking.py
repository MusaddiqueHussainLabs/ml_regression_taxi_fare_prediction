import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from urllib.parse import urlparse
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
            if experiment.name == "taxi_fare_prediction_v1":
                return experiment

        # Provide an Experiment description that will appear in the UI
        experiment_description = (
            "Taxi fare prediction experiment"
        )

        # Provide searchable tags that define characteristics of the Runs that
        # will be in this Experiment
        experiment_tags = {
            "project": "taxi_fare_prediction_v1",
            "store_dept": "Regression",
            "team": "regression-ml",
            "data_source": "NYC Taxi Dataset",
            "author": "MusaddiqueHussain Jahagirdar",
            "mlflow.note.content": experiment_description,
        }

        try:
            # Create the Experiment, providing a unique name
            taxi_fare_experiment = client.create_experiment(
                name="taxi_fare_prediction_v1", tags=experiment_tags
            )
        except mlflow.exceptions.RestException as e:
            print(f"MLflow API request failed with BAD_REQUEST: {e}")
            
        

        return taxi_fare_experiment
    
    def run_experiment(self, model_name, X_test, y_pred, model, params, metrics):

        # Sets the current active experiment to the "taxi_fare_prediction" experiment and
        # returns the Experiment metadata
        taxi_fare_experiment = mlflow.set_experiment("taxi_fare_prediction_v1")

        # Define a run name for this iteration of training.
        # If this is not set, a unique name will be auto-generated for your run.
        run_name = str(model_name).lower() + '_baseline'

        # Define an artifact path that the model will be saved to.
        artifact_path = run_name

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Initiate the MLflow run context
        with mlflow.start_run(run_name=run_name) as run:
            # Log the parameters used for the model fit
            mlflow.log_params(params)

            # Log the error metrics that were calculated during validation
            mlflow.log_metrics(metrics)

            # Log an instance of the trained model for later use
            signature = infer_signature(X_test, y_pred)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                
                mlflow.sklearn.log_model(
                    sk_model=model, input_example=X_test, artifact_path=artifact_path, signature=signature,
                    registered_model_name=run_name
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model, input_example=X_test, artifact_path=artifact_path, signature=signature
                )

            
