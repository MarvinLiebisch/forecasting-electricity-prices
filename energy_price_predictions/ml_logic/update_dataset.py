from entsoe import EntsoePandasClient
import pandas as pd
from datetime import date, timedelta
import subprocess
import pandas as pd

def get_git_root():
    git_root = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
    stdout, _ = git_root.communicate()
    return stdout.decode().strip()

def get_data_from_entsoe(api_key, country_code, start, end):

    client = EntsoePandasClient(api_key=api_key)

    df_day_ahead_prices = client.query_day_ahead_prices(country_code, start=start,end=end)
    df_day_ahead_prices_df = pd.DataFrame(df_day_ahead_prices, columns=['price_day_ahead'])
    df_day_ahead_prices_df_grouped = df_day_ahead_prices_df.resample('1H').mean()

    df_generation = client.query_generation(country_code, start=start, end=end, psr_type=None)
    df_generation.drop(('Hydro Pumped Storage', 'Actual Aggregated'), axis = 1, inplace=True)
    df_generation_grouped = df_generation.resample('1H').mean()
    df_merged = df_day_ahead_prices_df_grouped.join(df_generation_grouped.droplevel(1, axis='columns'))

    df_load = client.query_load(country_code, start=start, end=end)
    df_load_grouped = df_load.resample('1H').mean()
    df_merged_1 = df_merged.join(df_load_grouped)

    df_load_forecast = client.query_load_forecast(country_code, start=start, end=end)
    df_load_forecast_grouped = df_load_forecast.resample('1H').mean()
    df_merged_2 = df_merged_1.join(df_load_forecast_grouped)

    df_gen_forecast = client.query_generation_forecast(country_code, start=start, end=end)
    df_gen_forecast_grouped = df_gen_forecast.resample('1H').mean()
    df_gen_forecast_grouped.rename(columns={'Actual Aggregated': 'generation_forecast'}, inplace=True)
    df_merged_3 = df_merged_2.join(df_gen_forecast_grouped[['generation_forecast']])

    df_solar_wind_forecast = client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)
    df_solar_wind_forecast_grouped = df_solar_wind_forecast.resample('1H').mean()
    df_merged_4 = df_merged_3.join(df_solar_wind_forecast_grouped, rsuffix ='_forecast' )
    df_final = df_merged_4[:-1]

    return df_final

def filter_data(df):

    df.rename(columns={
        'Unnamed: 0' : 'time',
        'Biomass' : 'generation_biomass',
        'Fossil Brown coal/Lignite' : 'generation_fossil_brown_coal/lignite',
        'Fossil Gas' : 'generation_fossil_gas',
        'Fossil Hard coal' : 'generation_fossil_hard_coal',
        'Fossil Oil' : 'generation_fossil_oil',
        'Hydro Pumped Storage' : 'generation_hydro_pumped_storage_consumption',
        'Hydro Run-of-river and poundage' : 'generation_hydro_run-of-river_and_poundage',
        'Hydro Water Reservoir' : 'generation_hydro_water_reservoir',
        'Nuclear' : 'generation_nuclear',
        'Other' : 'generation_other',
        'Other renewable' : 'generation_other_renewable',
        'Solar' : 'generation_solar',
        'Waste' : 'generation_waste',
        'Wind Onshore' : 'generation_wind_onshore',
        'Actual Load' : 'total_load_actual',
        'Forecasted Load' : 'total_load_actual_forecast',
        'Solar_forecast' : 'forecast_solar_day_ahead',
        'Wind Onshore_forecast' : 'forecast_wind_onshore_day_ahead'
    }, inplace=True)


    return df[[
    'generation_fossil_hard_coal',
    'generation_fossil_gas',
    'generation_fossil_brown_coal/lignite',
    'total_load_actual',
    'total_load_actual_forecast',
    'generation_other_renewable',
    'generation_waste',
    'generation_fossil_oil',
    'generation_hydro_run-of-river_and_poundage',
    'generation_wind_onshore',
    'forecast_wind_onshore_day_ahead',
    'generation_hydro_pumped_storage_consumption','price_day_ahead']]


def update_clean_data(api_key, country_code):

    git_root = get_git_root()
    full_path = git_root + '/raw_data/'

    df_old = pd.read_csv(full_path + '/filtered_data.csv')
    df_old.index = pd.to_datetime(df_old['Unnamed: 0'])
    df_old.drop(columns=['Unnamed: 0'], inplace = True)

    today = date.today()
    start = pd.Timestamp((df_old.index.max()+timedelta(+1)).strftime("%Y%m%d"), tz='Europe/Brussels')
    end = pd.Timestamp((today+timedelta(0)).strftime("%Y%m%d"), tz='Europe/Brussels')

    if start == end:
        return print("Already up-to-date")


    df_new = get_data_from_entsoe(api_key, country_code, start, end)
    df_new_filtered = filter_data(df_new)
    df_merged = pd.concat([df_old, df_new_filtered])
    df_merged.to_csv(full_path+"/filtered_data.csv")
    return print(f"""New data from {(df_old.index.max()+timedelta(+1)).strftime('%d-%m-%y')} to
                 {(today+timedelta(-1)).strftime('%d-%m-%y')} imported and saved to filtered_data.csv.""")


#update_clean_data("API KEY HERE", "ES")
