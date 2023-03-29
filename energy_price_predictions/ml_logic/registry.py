import glob
import os
import time
import pickle
from tensorflow import keras
from tensorflow.keras import models
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_MODEL_NAME = "electricitypriceprediction"
MLFLOW_TRACKING_URI = "https://mlflow.lewagon.ai"
MLFLOW_EXPERIMENT = "electricitypriceprediction_experiment"


@st.cache(allow_output_mutation=True)
def load_model_cache(source='local'):
    if source == 'mlflow':
        model = load_model()
    elif source == 'local':
        model = models.load_model(f'models/electricitypriceprediction.h5')
    print(model.summary())
    return model


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics on mlflow
    """
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        mlflow.log_metrics(metrics)


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model on mlflow
    """

    mlflow.tensorflow.log_model(model=model,
                            artifact_path="model",
                            registered_model_name=MLFLOW_MODEL_NAME,
                            )



def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model or from MLFLOW (by "stage")
    Return None (but do not Raise) if no model found

    """

    # load model from mlflow
    model = None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
        model_uri = model_versions[0].source
        assert model_uri is not None
    except:
        return None

    model = mlflow.tensorflow.load_model(model_uri=model_uri)

    return model



def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from current_stage stage to new_stage and archive the existing model in new_stage
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    return None


def mlflow_run(func):
    """Generic function to log params and results to mlflow along with tensorflow autologging

    Args:
        func (function): Function you want to run within mlflow run
        params (dict, optional): Params to add to the run in mlflow. Defaults to None.
        context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)
        return results
    return wrapper
