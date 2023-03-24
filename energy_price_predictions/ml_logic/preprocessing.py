import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from data_import import import_merged_data


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
            ("categorical", one_hot_encoder, features_categorical)
        ],
        remainder = treat_remainder)

    return column_transformer.fit_transform(X)


def standardized_data(df, numerical_features=['generation_fossil_hard_coal',
                                              'generation_fossil_gas',
                                              'generation_fossil_brown_coal/lignite',
                                              'total_load_forecast',
                                              'total_load_actual',
                                              'generation_other_renewable',
                                              'generation_waste',
                                              'generation_fossil_oil',
                                              'generation_hydro_run-of-river_and_poundage',
                                              'generation_wind_onshore',
                                              'forecast_wind_onshore_day_ahead',
                                              'generation_hydro_pumped_storage_consumption'],
                    categorical_features=[],
                    target_variable='price_day_ahead'):
    """
    Perform preprocessing on the input data using StandardScaler for numerical features
    and OneHotEncoder for categorical features. Returns the preprocessed data and target variable.

    Parameters:
        df (pd.DataFrame): The input dataframe
        numerical_features (list): A list of numerical feature names
        categorical_features (list): A list of categorical feature names
        target_variable (str): The name of the target variable column

    Returns:
        X (np.array): The preprocessed data
        y (np.array): The target variable
    """
    preprocessing = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    X = preprocessing.fit_transform(df.drop(target_variable, axis=1))
    y = df[target_variable]

    return X, y



def normalize_dataset(df, numerical_features=['generation_fossil_hard_coal',
                      'generation_fossil_gas',
                      'generation_fossil_brown_coal/lignite',
                      'total_load_forecast',
                      'total_load_actual',
                      'generation_other_renewable',
                      'generation_waste',
                      'generation_fossil_oil',
                      'generation_hydro_run-of-river_and_poundage',
                      'generation_wind_onshore',
                      'forecast_wind_onshore_day_ahead',
                      'generation_hydro_pumped_storage_consumption'],
                    categorical_features=[],
                    target_variable='price_day_ahead'):
    """
    Perform preprocessing on the input data using MinMaxScaler for numerical features
    and OneHotEncoder for categorical features. Returns the preprocessed data and target variable.

    Parameters:
        df (pd.DataFrame): The input dataframe
        numerical_features (list): A list of numerical feature names
        categorical_features (list): A list of categorical feature names
        target_variable (str): The name of the target variable column

    Returns:
        X (np.array): The preprocessed data
        y (np.array): The target variable
    """
    preprocessing = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    X = preprocessing.fit_transform(df.drop(target_variable, axis=1))
    y = df[target_variable]

    return X, y

# you can call the function like this:
# X, y = normalize_dataset(df)
