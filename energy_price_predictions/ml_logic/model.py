import sys
import streamlit as st
import time
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from typing import Tuple
import numpy as np
import random

from energy_price_predictions.ml_logic.data_import import get_git_root
from energy_price_predictions.ml_logic.registry import load_model





# def sequence_data(X, y,
#                   n_observation_X, n_observation_y,
#                   n_sequence_train,n_sequence_val,n_sequence_test,
#                   val_cutoff,test_cutoff):


#     sample_list_train = list(range(0, int(len(X)*val_cutoff-n_observation_y-n_observation_X)))
#     sample_list_val = list(range(int(len(X)*val_cutoff),int(len(X)*test_cutoff)))
#     sample_list_test= list(range(int(len(X)*test_cutoff),int(len(X)-n_observation_y-n_observation_X)))

#     random.shuffle(sample_list_train)
#     random.shuffle(sample_list_val)
#     random.shuffle(sample_list_test)

#     X_train=np.zeros((n_sequence_train, n_observation_X, X.shape[1]))
#     X_val=np.zeros((n_sequence_val, n_observation_X, X.shape[1]))
#     X_test=np.zeros((n_sequence_test, n_observation_X, X.shape[1]))

#     y_train=np.zeros((n_sequence_train, n_observation_y, 1))
#     y_val=np.zeros((n_sequence_val, n_observation_y, 1))
#     y_test=np.zeros((n_sequence_test, n_observation_y, 1))


#     def create_sequence(X_,y_,sample_list,n_sequence):
#         index=0
#         for i in sample_list[0:n_sequence]:
#             X_[index] = X.iloc[i:i + n_observation_X].values
#             y_[index]= y.iloc[i + n_observation_X:i + n_observation_X + n_observation_y].values
#             index=index+1
#         return X_, y_

#     X_train, y_train = create_sequence(X_train,y_train,sample_list_train,n_sequence_train)
#     X_val, y_val = create_sequence(X_val,y_val,sample_list_val,n_sequence_val)
#     X_test, y_test = create_sequence(X_test,y_test,sample_list_test,n_sequence_test)

#     return X_train, X_val, X_test, y_train, y_val,y_test


def initialize_model(input_shape, output_shape=24) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    model = Sequential()
    model.add(layers.LSTM(units=64, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(units=64, return_sequences=True))
    model.add(layers.LSTM(units=64, return_sequences=False))
    model.add(layers.Dense(output_shape, activation='linear'))  # Output layer with 24 neurons (one for each hour)
    # model = Sequential([
    #     layers.SimpleRNN(units=64, input_shape=(None, input_shape), return_sequences=True),
    #     layers.Dense(24, activation='linear')
    # ])
    return model


def compile_model(model: Model, learning_rate=0.01) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Nadam(learning_rate=learning_rate) #change the optimizer and right all the default value
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=16,
                patience=15,
                epochs=1000,
                validation_data=None, # overrides validation_split
                validation_split=0.3) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    es = EarlyStopping(patience=patience, restore_best_weights=True)

    history = model.fit(X,
                        y,
                        validation_data=validation_data,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1,
                        shuffle=False)

    return model, history


def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=16) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    if model is None:
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True)

    return metrics



if __name__ == "__main__":
    pass
    # model = load_model('gru_model.h5')
    # model.summary()
