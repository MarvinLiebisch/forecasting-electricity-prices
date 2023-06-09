import numpy as np
import pandas as pd
import streamlit as st
import random

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

from energy_price_predictions.ml_logic.data_import import import_merged_data


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def run_pipeline(df, numerical_scaler = "min_max_scaler", time_features = True,
                 shift_value = 24, max_categories = 10, treat_remainder = 'drop'):

    ''' Runs provided features through preprocessing pipeline.
        Object columns are one_hot_encoded.
        Numeric columns are scaled with specified scaler.
        Cyclical features are SinCos transformed.

        Parameters:
                df (DataFrame): df of features with datetime index and target columng: 'price_day_ahead'
                numerical_scaler (str): [min_max_scaler, standard_scaler]
                time_features (bool): generates and transform time features
                shift_value(int): periods to shift y
                max_categories (int): maximum  number of categories per feature (if more -> remainder)
                treat_remainder ('drop', 'passthrough', 'min_max_scaler'): defines how uncaptured columns are treated (default -> dropped)

        Returns:
                np.array
    '''

    # Create y-shift feature
    df['price_day_ahead_shifted'] = df['price_day_ahead'].shift(shift_value)

    # Define target
    y = df['price_day_ahead']

    # Remove target from features
    X = df.drop(columns=['price_day_ahead'])



    # Define the scalers
    if numerical_scaler == "min_max_scaler":
        scaler_to_use =  MinMaxScaler()

    elif numerical_scaler == "standard_scaler":
        scaler_to_use = StandardScaler()

    else:
        return print("Error: numerical_scaler must be 'min_max_scaler' or 'standard_scaler'")

    one_hot_encoder = OneHotEncoder()



    # Find numeric an categorical features to transform
    features_categorical_nunique = X.select_dtypes(include='object').nunique()
    features_categorical = list(features_categorical_nunique[features_categorical_nunique <= max_categories].index)

    # Transform features including engineered cyclical features
    if time_features == True:
        X['hour'] = X.index.hour
        X['month'] = X.index.month
        X['day_of_week'] = X.index.dayofweek
        X['day_of_year'] = X.index.dayofyear

        preprocessor = ColumnTransformer(
            transformers=[
                (f"numeric_{scaler_to_use}", scaler_to_use, make_column_selector(dtype_include=np.number)),
                ("categorical", one_hot_encoder, features_categorical),
                ("month_sin", sin_transformer(12), ["month"]),
                ("month_cos", cos_transformer(12), ["month"]),
                ("weekday_sin", sin_transformer(7), ["day_of_week"]),
                ("weekday_cos", cos_transformer(7), ["day_of_week"]),
                ("hour_sin", sin_transformer(24), ["hour"]),
                ("hour_cos", cos_transformer(24), ["hour"]),
                ("dayofyear_sin", sin_transformer(365), ["day_of_year"]),
                ("dayofyear_cos", cos_transformer(365), ["day_of_year"]),
            ],
            remainder = treat_remainder)

    # Transform only features from dataset
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                (f"numeric_{scaler_to_use}", scaler_to_use, make_column_selector(dtype_include=np.number)),
                ("categorical", one_hot_encoder, features_categorical),
            ],
            remainder = treat_remainder)
    ##########################
    ### REAPLCE WITH MODEL ###

    # # Define the RNN model
    rnn_model = Sequential([
        SimpleRNN(units=64, input_shape=(None, X.shape[1]), return_sequences=True),
        Dense(24, activation='linear')
    ])

    rnn_model.compile(optimizer='adam', loss='mse')

    ### REPLACE WITH MODEL ###
    ##########################


    # Combine preprocessing and model into a single pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', rnn_model)
    ])

    ##############################
    ### NO FITTING YET
    # # Train the pipeline on your data
    #model_pipeline.fit(X, y)
    ##############################

    # Return the model pipeline
    return model_pipeline

    # model_pipeline.fit(X)

@st.cache(allow_output_mutation=True)
def get_preprocessed_data(dataframe, y):
    n_observation_X = 24 * 7*4  # For example, a week of data for the sequence
    n_observation_y = 24 # We would like to forecast the 24 prices of the next day during the auction of today
    n_sequence = 200
    preprocessor  = run_pipeline(dataframe)
    print(preprocessor)
    X_preprocessed = pd.DataFrame(preprocessor.fit_transform(dataframe))
    sample_list = list(range(0, len(X_preprocessed)))
    X__=np.zeros((n_sequence, n_observation_X, dataframe.shape[1]))
    y__=np.zeros((n_sequence, n_observation_y, 1))
    def create_sequence(X_,y_,sample_list,n_sequence):
        index=0
        for i in sample_list[0:n_sequence]:
            X_[index] = X_preprocessed.iloc[i:i + n_observation_X].values
            y_[index]= y.iloc[i + n_observation_X:i + n_observation_X + n_observation_y].values
            index=index+1
        return X_, y_
    return create_sequence(X__, y__, sample_list, n_sequence)


if __name__ == "__main__":
    df = import_merged_data()

    # Train the RNN pipeline on your data
    model = run_pipeline(df)
    print(model)
