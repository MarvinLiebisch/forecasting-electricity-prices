import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0,'/home/rahmah/code/marvinliebisch/forecasting-electricity-prices')

from energy_price_predictions.ml_logic.data_import import *
from energy_price_predictions.ml_logic.preprocessing import *
from energy_price_predictions.ml_logic.registry import *
from energy_price_predictions.ml_logic.model import *


def preprocess():
    df = import_merged_data()
    df = df.dropna()
    X, y = normalize_dataset(df)
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    n_observation_X=24 * 7*4  # For example, a week of data for the sequence
    n_observation_y=24 # We would like to forecast the 24 prices of the next day during the auction of today
    n_sequence_train=60
    n_sequence_val=5
    n_sequence_test=5
    val_cutoff=0.8
    test_cutoff=0.9

    X_train, X_val, X_test, y_train, y_val,y_test = sequence_data(X, y,
                  n_observation_X, n_observation_y,
                  n_sequence_train,n_sequence_val,n_sequence_test,
                  val_cutoff,test_cutoff)
    return X_train, X_val, X_test, y_train, y_val,y_test

@mlflow_run
def train(X_train, y_train, X_val, y_val) -> float:
    """
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as float
    """

    # Train model using `model.py`
    # model = load_model()
    # if model is None:
    model = initialize_model(input_shape=X_train.shape[1:], output_shape=y_train.shape[1])
    model = compile_model(model)
    model, history = train_model(model, X_train, y_train,
                                 validation_data=(X_val, y_val))

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=len(X_train),
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    # The latest model should be moved to staging
    mlflow_transition_model(current_stage="None", new_stage="Staging")

    return val_mae


@mlflow_run
def evaluate(X_new, y_new, stage: str = "Production") -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return mae as float
    """

    model = load_model(stage=stage)
    assert model is not None

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
    )

    save_results(params=params, metrics=metrics_dict)

    return mae
