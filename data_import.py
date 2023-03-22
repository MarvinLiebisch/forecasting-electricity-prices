import pandas as pd
import os


def import_data(dataset):
    '''Imports sepcified dataset fomr raw_data directory.

        Parameters:
                dataset (str): energy_data or weather_features

        Returns:
                DataFrame
    '''

    path = os.getcwd()
    full_path = path + '/raw_data/' + dataset + '.csv'
    print(f"Importing {dataset} data from {full_path}...")
    df_raw = pd.read_csv(full_path)

    return df_raw


def import_clean_energy_data(dropNA=True):
    '''Returns clean energy data.

        Parameters:
                dropNA (bool): Default is True (Dropping NA rows)

        Returns:
                DataFrame
    '''
    df_energy = import_data('energy_dataset')
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
    df_energy['time'] = df_energy['time'].dt.tz_convert('Europe/Madrid')
    df_energy = df_energy.set_index('time')
    df_energy = df_energy.drop(columns=['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead', 'generation marine', 'generation fossil oil shale', 'generation fossil peat', 'generation geothermal', 'generation fossil coal-derived gas', 'generation wind offshore'])
    df_energy.columns = df_energy.columns.str.replace(' ', '_')

    if dropNA == True:
        df_energy = df_energy.dropna()

    return df_energy


def import_clean_weather_data():
    '''Returns clean weather data.

        Parameters:
                None

        Returns:
                DataFrame
    '''
    df_weather = import_data('weather_features')
    df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
    df_weather = df_weather.rename(columns={'dt_iso': 'time'})
    df_weather = df_weather.set_index('time')

    return df_weather
