import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from data_import import import_merged_data


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def run_pipeline(X, max_categories = 10, treat_remainder = 'drop'):
    ''' Runs provided features through preprocessing pipeline.
        Object columns are one_hot_encoded.
        Numeric columns are MinMaxScaled.

        Parameters:
                X (DataFrame): features
                max_categories (int): maximum  number of categories per feature (if more -> remainder)
                treat_remainder ('drop', 'passthrough', 'min_max_scaler'): defines how uncaptured columns are treated (default -> dropped)
        Returns:
                np.array
    '''

    one_hot_encoder = OneHotEncoder()
    min_max_scaler =  MinMaxScaler()

    features_categorical_nunique = X.select_dtypes(include='object').nunique()
    features_categorical = list(features_categorical_nunique[features_categorical_nunique <= max_categories].index)


    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric_min_max", min_max_scaler, make_column_selector(dtype_include=np.number)),
            ("categorical", one_hot_encoder, features_categorical),
            ("month_sin", sin_transformer(12), ["month"]),
            ("month_cos", cos_transformer(12), ["month"]),
            ("weekday_sin", sin_transformer(7), ["day_of_week"]),
            ("weekday_cos", cos_transformer(7), ["day_of_week"]),
            ("hour_sin", sin_transformer(24), ["hour"]),
            ("hour_cos", cos_transformer(24), ["hour"]),
        ],
        remainder = treat_remainder)

    return column_transformer.fit_transform(X)
