
import streamlit as st
import requests
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import sys
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_git_root():
    git_root = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
    stdout, _ = git_root.communicate()
    return stdout.decode().strip()
git_root = get_git_root()
git_path = git_root

sys.path.insert(0,git_path)

from energy_price_predictions.ml_logic.data_import import import_merged_data


@st.cache
def load_data():
    df = import_merged_data()
    df_price = df[['price_day_ahead']]
    df_price['7D_SMA'] = df_price['price_day_ahead'].rolling(24*7).mean()
    df_price['30D_SMA'] = df_price['price_day_ahead'].rolling(24*30).mean()
    df_price.dropna(inplace=True)
    return df_price


df_price = load_data()

'''
# Day-Ahead Energy Price Prediction Model
'''

'''


## Historical Auction Prices

'''
#@st.cache
def plot_1(df_price):
    fig = px.line(df_price, x=df_price.index, y=['price_day_ahead', '30D_SMA'], color_discrete_sequence=['rgba(200, 200, 200, 0.1)',
                                                                                                        'rgba(255, 255, 255, 1)'])


    fig_copy = fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    newnames = {'30D_SMA':'30D SMA', 'price_day_ahead': 'Hourly Price'}
    fig_copy.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                        )
                    )

    fig_copy.update_layout(
        title="Average Day-ahead Auction Prices",
        xaxis_title="",
        yaxis_title="Price",
        legend_title=None,
        font=dict(
            family="Arial",
            size=14,
            color="white"
        )
    )
    return fig_copy

fig1 = plot_1(df_price)
st.plotly_chart(fig1)

'''
## Hourly Auction Data
'''

# Generate placeholder data
date_range = pd.date_range("2022-01-01", "2022-12-31", freq="D")
df = pd.DataFrame()
for i in range(24):
    df[f"Series {i+1}"] = pd.Series(np.random.normal(100, 10, len(date_range)), index=date_range)

option = st.selectbox(
    'Hour to display',
    (df.columns))

st.write('You selected:', option)




# '''
# ## Please enter Feature data
# '''
# import  datetime


# with st.form(key='Adress'):
#     c1, c2, c3, c4 = st.columns(4)
#     with c1:
#         pick_lat = st.text_input("Pickup Latitude", value=40.758896)
#         d = st.date_input("Date", datetime.date(2019, 7, 6))
#     with c2:
#         pick_lon = st.text_input('Pickup Longtitude', value=-73.985130)
#         t = st.time_input('Time', datetime.time(8, 45))
#     with c3:
#         drop_lat = st.text_input('Dropoff Latitude', value=40.748817)
#         p = st.selectbox('Passengers', (1,2,3,4,5,6,7,8))
#     with c4:
#         drop_lon = st.text_input('Dropoff Longtitude', value=-73.985428)
#         submitButton = st.form_submit_button(label = 'Get Prediction')

# '''
# ## Price Prediction
# '''

# if submitButton:
#     dt_input = datetime.datetime.combine(d, t)


#     req_dict = {'pickup_datetime': str(dt_input), 'pickup_longitude': pick_lon, 'pickup_latitude': pick_lat,
#                 'dropoff_longitude': drop_lon, 'dropoff_latitude': drop_lat, 'passenger_count': p}

#     api_address = ""

#     req = requests.get(api_address, params=req_dict).json()

#     st.markdown('### The estimated fare is: ' + str(round(req['fare'],2)) + ' USD')
#     pick_lat = float(pick_lat)
#     pick_lon = float(pick_lon)
#     drop_lat = float(drop_lat)
#     drop_lon = float(drop_lon)

#     #map_df = pd.DataFrame({'lat': [pick_lat, drop_lat], 'lon': [pick_lon, drop_lon]})
#     #map_df['color'] = [255, 140, 0]
#     #st.map(map_df)
#     df_test = pd.DataFrame({'Name': 'Start to End' ,'color': [[57, 255, 20]], 'path':[[[pick_lon, pick_lat],[drop_lon, drop_lat]]]})


#     view_state = pdk.ViewState(
#         latitude=pick_lat,
#         longitude=pick_lon,
#         zoom=11
#     )

#     layer = pdk.Layer(
#         type='PathLayer',
#         data=df_test,
#         pickable=True,
#         get_color= 'color',
#         width_scale=20,
#         width_min_pixels=2,
#         get_path='path',
#         get_width=5
#     )

#     r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"})

#     st.pydeck_chart(r)


#streamlit run app.py
