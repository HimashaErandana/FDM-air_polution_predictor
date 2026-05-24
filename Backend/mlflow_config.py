import mlflow

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "AirPollutionPredictor"
MODEL_NAME = "RandomForestRiskClassifier"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow
