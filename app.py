import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.express as px
import requests
import json
import subprocess
import sys

from energy_price_predictions.ml_logic.data_import import *
# from energy_price_predictions.ml_logic.registry import *
# from energy_price_predictions.ml_logic.preprocessing_prod import *

#API_URL = "https://electricitypricepredictions-nrgmmiocwa-ew.a.run.app/predict"


# Currency Data API

# app_id = st.secrets["CUR_API_KEY"]
# url = f"https://openexchangerates.org/api/latest.json?app_id={app_id}&prettyprint=false&show_alternative=false"

# headers = {"accept": "application/json"}

# response_cur = requests.get(url, headers=headers)
# cur_data = response_cur.json()
# EURUSD = 1/cur_data['rates']['EUR']
# EURGBP = cur_data['rates']['GBP']*EURUSD
# RATE = {
#     'EUR': 1,
#     'USD': EURUSD,
#     'GBP': EURGBP,


RATE = {
    'EUR': 1,
    'USD': 1.08,
    'GBP': 0.88,
}


def create_plot_historical_data(df, colors, is_sma=True):
    xaxis=dict(
        title=None,
        showgrid=False,
        showline=False,
        zeroline=False,
        rangeselector=dict(
            buttons=list([
                dict(count=24, label="24 hour", step="hour", stepmode="backward"),
                dict(count=3, label="3 days", step="day", stepmode="backward"),
                dict(count=7, label="1 week", step="day", stepmode="backward"),
                dict(count=1, label="1 month", step="month", stepmode="backward"),
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(step="all"),
            ])
        )
    )
    yaxis=dict(
        title=f'Price day-ahead per MWh (in {currency})',
        showgrid=True,
        showline=False,
        zeroline=False,
    )
    legend=dict(
        yanchor="bottom",
        y=-0.3,
        xanchor="left",
        x=0,
        title=None,
        orientation='h',
    )

    fig = px.line(df,
        x='time',
        y=['price_day_ahead', 'price_day_ahead_prediction', '30D_SMA'] if is_sma else ['price_day_ahead', 'price_day_ahead_prediction'],
        #width=15,
    )

    fig.update_layout(
        xaxis=xaxis,
        yaxis=yaxis,
        legend=legend,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        xaxis_rangeselector_font_color='black',
        xaxis_rangeselector_activecolor='gray',
        xaxis_rangeselector_bgcolor='silver',
    )

    # Update legend name
    newnames = {
        'price_day_ahead': 'Real price',
        'price_day_ahead_prediction': 'Predicted Price',
    }
    colors_ = {
        'price_day_ahead': colors[0],
        'price_day_ahead_prediction': colors[1],
    }
    line_width = {
        'price_day_ahead': 1,
        'price_day_ahead_prediction': 1,
    }
    if is_sma:
        newnames['30D_SMA'] = 'Real price 30D_SMA'
        colors_['30D_SMA'] = colors[2]
        line_width['30D_SMA'] = 1
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name]),
                                        line = dict(color=colors_[t.name], width=line_width[t.name]),
                                        )
                    )
    return fig


def create_plot_hourly(df, colors, columns, width):
    xaxis=dict(
        title=None,
        showgrid=False,
        showline=False,
        zeroline=False,
        rangeselector=dict(
            buttons=list([
                dict(count=24, label="24 hour", step="hour", stepmode="backward"),
                dict(count=3, label="3 days", step="day", stepmode="backward"),
                dict(count=7, label="1 week", step="day", stepmode="backward"),
                dict(count=1, label="1 month", step="month", stepmode="backward"),
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(step="all"),
            ])
        )
    )


    yaxis=dict(
        title=f'Price day-ahead per MWh (in {currency})',
        showgrid=True,
        showline=False,
        zeroline=False,
    )

    legend=dict(
        yanchor="bottom",
        y=-0.3,
        xanchor="left",
        x=0,
        title=None,
        orientation='h',
    )

    fig = px.line(df,
            x='time',
            y=columns,
            width=15,
        )

    fig.update_layout(
        xaxis=xaxis,
        yaxis=yaxis,
        legend=legend,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        xaxis_rangeselector_font_color='black',
        xaxis_rangeselector_activecolor='gray',
        xaxis_rangeselector_bgcolor='silver',
    )
    # Update legend name
    colors_ = {col: colors[idx] for idx,col in enumerate(columns)}
    line_width = {col: width for idx,col in enumerate(columns)}
    fig.for_each_trace(lambda t: t.update(line = dict(color=colors_[t.name], width=line_width[t.name])))
    return fig

### START ###
# ==================

st.set_page_config(page_title="Electricy Price Prediction") #page_icon=img,
### Add our logo to the top of the page
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')

with col2:
    st.image("https://i.ibb.co/1RGqJKL/Neural-Energy-white-on-streamlit.png", width=300)

with col3:
    st.write(' ')
####
'''
#
#
'''
st.markdown('# Electricity Price Prediction :zap:')

# Retrieve data
data = import_final_result_cache()
df = data.reset_index()
# y = data[['price_day_ahead']]

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox('Select country',
        ['ES', 'FR 🚧', 'DE 🚧', 'UK 🚧']
    )
with col2:
    currency = st.selectbox('Select price currency',
        ['EUR', 'USD', 'GBP']
    )

st.markdown("***")

df['price_day_ahead'] = RATE[currency] * df['price_day_ahead']
df['price_day_ahead_prediction'] = RATE[currency] * df['price_day_ahead_prediction']

# Table 1
# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

def color_template(val):
    color = 'green' if float(val.split(' ')[1]) >= 0 else 'red'
    return f'color: {color}'

st.markdown(hide_table_row_index, unsafe_allow_html=True)
today = df['time'].iloc[-1].date()
yday = today - timedelta(1)
twoday = today - timedelta(2)
lweek = today - timedelta(7)
st.markdown(f'''
### Day-ahead Price Predictions (as of {str(today.strftime("%d/%m/%Y"))})
''')
### Day-ahead Price Predictions (as of {str(today)})

table_yday = df[df['time'].apply(lambda x: x.date()) == yday][['time', 'price_day_ahead']].drop_duplicates('time')
table_2yday = df[df['time'].apply(lambda x: x.date()) == twoday][['time', 'price_day_ahead']].drop_duplicates('time')
table_today = df[df['time'].apply(lambda x: x.date()) == today][['time', 'price_day_ahead_prediction']].drop_duplicates('time')
table_lweek = df[df['time'].apply(lambda x: x.date()) == lweek][['time', 'price_day_ahead']].drop_duplicates('time')
table_yday['Hour'] = table_yday['time'].apply(lambda x: x.hour).astype(str)
table_today['Hour'] = table_today['time'].apply(lambda x: x.hour).astype(str)
table_lweek['Hour'] = table_lweek['time'].apply(lambda x: x.hour).astype(str)
table_yday['price_day_ahead_yday'] = table_yday['price_day_ahead']
table_today['price_day_ahead_today'] = table_today['price_day_ahead_prediction']
table_lweek['price_day_ahead_lweek'] = table_lweek['price_day_ahead']
table_yday.set_index('Hour', inplace=True)
table_today.set_index('Hour', inplace=True)
table_lweek.set_index('Hour', inplace=True)
table_diff_day = table_today[['price_day_ahead_today']].join(table_yday[['price_day_ahead_yday']], on='Hour')
table_diff_week = table_today[['price_day_ahead_today']].join(table_lweek[['price_day_ahead_lweek']], on='Hour')
table_today['Change vs. Today (%)'] = (table_diff_day['price_day_ahead_today']/table_diff_day['price_day_ahead_yday']-1)*100
table_today['Change vs. Last Week (%)'] = (table_diff_week['price_day_ahead_today']/table_diff_week['price_day_ahead_lweek']-1)*100

########################################################################
dm_high_hour = table_today['Change vs. Today (%)'].idxmax()
dm_high = table_today['price_day_ahead_prediction'][dm_high_hour]
dm_high_change = table_today['Change vs. Today (%)'][dm_high_hour]
dm_low_hour = table_today['Change vs. Today (%)'].idxmin()
dm_low = table_today['price_day_ahead_prediction'][dm_low_hour]
dm_low_change= table_today['Change vs. Today (%)'][dm_low_hour]
wm_high_hour = table_today['Change vs. Last Week (%)'].idxmax()
wm_high = table_today['price_day_ahead_prediction'][wm_high_hour]
wm_high_change = table_today['Change vs. Last Week (%)'][wm_high_hour]
wm_low_hour = table_today['Change vs. Last Week (%)'].idxmin()
wm_low = table_today['price_day_ahead_prediction'][wm_low_hour]
wm_low_change = table_today['Change vs. Last Week (%)'][wm_low_hour]
########################################################################

table_today['Change vs. Today (%)'] = table_today['Change vs. Today (%)'].apply(lambda x: f'▲ {round(x,2)}' if x > 0 else f'▼ {round(x,2)}')
table_today['Change vs. Last Week (%)'] = table_today['Change vs. Last Week (%)'].apply(lambda x: f'▲ {round(x,2)}' if x > 0 else f'▼ {round(x,2)}')
table_today[f'Predicted Price ({currency})'] = table_today['price_day_ahead_prediction'].apply(lambda x: f'{round(x, 2)}')
table_today_chart = table_today.reset_index()[['Hour', f'Predicted Price ({currency})', 'Change vs. Today (%)', 'Change vs. Last Week (%)']]

col1, col2 = st.columns(2)
with col1:
    st.table(table_today_chart.iloc[0:-12].style.applymap(color_template, subset=['Change vs. Today (%)', 'Change vs. Last Week (%)']))
with col2:
    st.table(table_today_chart.iloc[12:24].style.applymap(color_template, subset=['Change vs. Today (%)', 'Change vs. Last Week (%)']))

# Ticker
# For below text boxes
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('#### Daily Movers')
    st.markdown(f"Hour {dm_high_hour}")
    st.metric(f'High (24 h)', round(dm_high,2),  f'{round(dm_high_change,1)} %')

with col2:
    st.markdown('#### ')
    st.markdown('#### ')
    st.markdown(f"Hour {dm_low_hour}")
    st.metric(f'Low (24 h)', round(dm_low,2),  f'{round(dm_low_change,1)} %')

with col3:
    st.markdown('#### Weekly Movers')
    st.markdown(f"Hour {wm_high_hour}")
    st.metric(f'High (7 d)', round(wm_high,2),  f'{round(wm_high_change,1)} %')

with col4:
    st.markdown('#### ')
    st.markdown('#### ')
    st.markdown(f"Hour {wm_low_hour}")
    st.metric(f'Low (7 d)', round(wm_low,2),  f'{round(wm_low_change,1)} %')


# Plot 1
'''
#
#
### Day-ahead Prices Min/Max/Mean
'''
df_plot1 = df
col1, col2 = st.columns(2)
with col1:
    selected_plot1 = st.selectbox('Select hour',
        [f'{i} ' for i in range(0,24)]
    )
with col2:
    selected_rolling_window = int(st.radio('Select day window',
        [14,30,60],
        horizontal=True,
    ))
# Filter based on selected hour
hour_plot1 = int(selected_plot1.split(' ')[0])
df_plot1 = df_plot1[df_plot1['time'].apply(lambda x: x.hour) == hour_plot1]

df_plot1['price_day_ahead'] = RATE[currency] * df_plot1['price_day_ahead']
df_plot1[f'Min {selected_rolling_window}d'] = df_plot1['price_day_ahead'].rolling(selected_rolling_window).min()
df_plot1[f'Mean {selected_rolling_window}d'] = df_plot1['price_day_ahead'].rolling(selected_rolling_window).mean()
df_plot1[f'Max {selected_rolling_window}d'] = df_plot1['price_day_ahead'].rolling(selected_rolling_window).max()

fig_plot1 = create_plot_hourly(df_plot1,
    colors = ['purple', 'green', 'blue'],
    columns = [f'Min {selected_rolling_window}d', f'Mean {selected_rolling_window}d', f'Max {selected_rolling_window}d'],
    width = 1.5,
)

st.plotly_chart(fig_plot1, theme="streamlit", use_container_width=True)

# Plot 2
'''
#
#
### Historical Prices vs. Predictions (overall)
'''
df['30D_SMA'] = df['price_day_ahead'].rolling(24*30).mean()
# This is for streamlit main page (chart)
# Properties of figure
fig = create_plot_historical_data(df, colors=['blue', 'green', 'white'], is_sma=True)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Plot 3
'''
#
#
### Historical Prices vs. Predictions (by hour)
'''
df_hour = df
col1, col2 = st.columns(2)
with col1:
    selected_hour = st.selectbox('Select hour ',
        [f'{i}' for i in range(0,24)]
    )
# Filter based on selected hour
if selected_hour != 'all':
    hour_value = int(selected_hour.split(' ')[0])
    df_hour = df_hour[df_hour['time'].apply(lambda x: x.hour) == hour_value]

df_hour['price_day_ahead'] = RATE[currency] * df_hour['price_day_ahead']

fig_hour = create_plot_historical_data(df_hour,

                                       colors=['green', 'blue'],
                                       is_sma=False)
                                       #colors=['rgba(180, 180, 180, 1)','rgba(255, 255, 255, 1)'],
st.plotly_chart(fig_hour, theme="streamlit", use_container_width=True)

st.text("")
st.markdown("***")
st.text("")

'''
#### :mailbox: Subscribe to our mailing list to receive updates and daily price information!
'''

contact_form = """
<form action="https://formsubmit.co/YOUREMAIL@EMAIL.COM" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <button type="submit">Send</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)

st.text("")
st.markdown("***")
st.text("")

'''
##### Disclaimer
"The information provided on this website is for educational and informational purposes only.
It is not intended to be and does not constitute financial advice.
NeuralEnergy does not make any representations or warranties about the accuracy or
completeness of the information provided on this website.
Any reliance you place on the information is strictly at your own risk.
NeuralEnergy will not be liable for any losses or damages in connection
with the use of this website or the information contained herein."
'''
