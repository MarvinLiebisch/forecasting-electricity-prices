import pandas as pd
import os
import subprocess

def get_git_root():
    git_root = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
    stdout, _ = git_root.communicate()
    return stdout.decode().strip()

def import_data(dataset):
    '''Imports sepcified dataset fomr raw_data directory.

        Parameters:
                dataset (str): energy_data or weather_features

        Returns:
                DataFrame
    '''

    
    
    git_root = get_git_root()
    full_path = git_root + '/raw_data/' + dataset + '.csv'
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


def import_merged_data(FeatureEngineering = True, TempMin = False, TempMax = False, WeatherIcon = False):
    '''Returns merged DataFrame containing the energy and weather data.

    Parameters:
            FeatureEngineering (bool): Default is True
            (Creates several time based features
            e.g. season, month, day of week, weekend, hour)

            TempMin, TempMax, WeatherIcon (bool): Default is False
            (Drops specified weather columns entirely)

    Returns:
            DataFrame
    '''

    df_energy = import_clean_energy_data(dropNA=True)
    df_weather = import_clean_weather_data()

    if FeatureEngineering == True:
        df_energy['hour'] = df_energy.index.hour
        df_energy['month'] = df_energy.index.month

        season_dict = {1: 'Winter',
                    2: 'Winter',
                    3: 'Spring',
                    4: 'Spring',
                    5: 'Spring',
                    6: 'Summer',
                    7: 'Summer',
                    8: 'Summer',
                    9: 'Fall',
                    10: 'Fall',
                    11: 'Fall',
                    12: 'Winter'}

        df_energy['season'] = df_energy.index.month.map(lambda x: season_dict[x])
        df_energy['weekend'] = (df_energy.index.dayofweek > 4).astype(int)
        df_energy['day_of_week'] = df_energy.index.dayofweek

    if TempMin == False:
        df_weather = df_weather.drop(columns=['temp_min'])
    if TempMax == False:
        df_weather = df_weather.drop(columns=['temp_max'])
    if WeatherIcon == False:
        df_weather = df_weather.drop(columns=['weather_icon'])

    df_weather['city_name'] = df_weather['city_name'].str.strip()
    city_array = df_weather['city_name'].unique()
    grouped = df_weather.groupby(df_weather.city_name)
    df_merged = df_energy

    for city in city_array:
        df_merged = df_merged.join(grouped
                                   .get_group(city)
                                   .drop(columns=['city_name'])
                                   .add_prefix(city.lower() + '_'))

    return df_merged

