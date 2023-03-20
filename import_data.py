import pandas as pd
import os


def import_data():

    path = os.getcwd()
    print(f"Importing data from {path}...")
    df = pd.read_csv(path + '/raw_data/energy_dataset.csv')

    return df
