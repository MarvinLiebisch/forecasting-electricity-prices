import pandas as pd
import os

def import_data():

    path = os.getcwd()
    print(f"Importing data from {path}...")
    df = pd.read_csv(path + '/raw_data/energy_dataset.csv')

    return df


def import_clean_data(dropNA=True):
    '''Returns cleaned energy data.

        Parameters:
                dropNA (bool): Default is True (Dropping NA rows)

        Returns:
                DataFrame
    '''

    df = import_data()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df = df.drop(columns=['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead', 'generation marine', 'generation fossil oil shale', 'generation fossil peat', 'generation geothermal', 'generation fossil coal-derived gas', 'generation wind offshore'])
    df.columns = df.columns.str.replace(' ', '_')

    if dropNA == True:
        df = df.dropna()

    return df
