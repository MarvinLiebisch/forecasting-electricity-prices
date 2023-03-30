from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
import random

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))




def run_pipeline(df, numerical_scaler = "min_max_scaler", time_features = True,
                shift_value = 24, max_categories = 0, treat_remainder = 'drop'
                ,initialization=True,numerical_features=['generation_fossil_hard_coal',
                                        'generation_fossil_gas',
                                        'generation_fossil_brown_coal/lignite',
                                        'total_load_actual',
                                        'generation_other_renewable',
                                        'generation_waste',
                                        'generation_fossil_oil',
                                        'generation_hydro_run-of-river_and_poundage',
                                        'generation_wind_onshore',
                                        'generation_hydro_pumped_storage_consumption']):

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



    # Define the scalers
    if numerical_scaler == "min_max_scaler":
        scaler_to_use =  MinMaxScaler()

    elif numerical_scaler == "standard_scaler":
        scaler_to_use = StandardScaler()

    else:
        return print("Error: numerical_scaler must be 'min_max_scaler' or 'standard_scaler'")

    one_hot_encoder = OneHotEncoder()


    # Find numeric an categorical features to transform
    features_categorical_nunique = df.select_dtypes(include='object').nunique()
    features_categorical = list(features_categorical_nunique[features_categorical_nunique <= max_categories].index)

    # Transform features including engineered cyclical features
    if time_features == True:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear

        preprocessor = ColumnTransformer(
            transformers=[
                (f"numeric_{scaler_to_use}", scaler_to_use, numerical_features),
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
                (f"numeric_{scaler_to_use}", scaler_to_use, numerical_features),
                ("categorical", one_hot_encoder, features_categorical),
            ],
            remainder = treat_remainder)
    ##########################
    ### REAPLCE WITH MODEL ###

    ##############################
    ### NO FITTING YET
    # # Train the pipeline on your data
    #model_pipeline.fit(X, y)
    ##############################

    # Return the model pipeline
    return preprocessor
