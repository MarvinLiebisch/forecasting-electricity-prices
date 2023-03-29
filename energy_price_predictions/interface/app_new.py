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
from energy_price_predictions.ml_logic.visualization import *
from energy_price_predictions.ml_logic.model import *
from energy_price_predictions.ml_logic.registry import *
from energy_price_predictions.ml_logic.preprocessing_prod import *

#API_URL = "https://electricitypricepredictions-nrgmmiocwa-ew.a.run.app/predict"
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
        title=f'price day ahead per MWh (in {currency})',
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
        y=['price_day_ahead', 'price_day_ahead_predicted', '30D_SMA'] if is_sma else ['price_day_ahead', 'price_day_ahead_predicted'],
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
    newnames = {
        'price_day_ahead': 'Real price',
        'price_day_ahead_predicted': 'Predicted price',
    }
    colors_ = {
        'price_day_ahead': colors[0],
        'price_day_ahead_predicted': colors[1],
    }
    line_width = {
        'price_day_ahead': 0.5,
        'price_day_ahead_predicted': 1,
    }
    if is_sma:
        newnames['30D_SMA'] = 'Real price 30D_SMA'
        colors_['30D_SMA'] = colors[2]
        line_width['30D_SMA'] = 2
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
    )
    yaxis=dict(
        title=f'price day ahead per MWh (in {currency})',
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
    st.image("https://i.ibb.co/1RGqJKL/Neural-Energy-white-on-streamlit.png", width=200)

with col3:
    st.write(' ')
####
'''
#
#
'''
st.markdown('# Electricity Price Prediction :zap:')

# Retrieve data
data = import_merged_data_cache()
print(data.shape)
df = data.reset_index()
y = data[['price_day_ahead']]

# preprocess data
preprocessor  = run_pipeline(data)
X_preprocessed = pd.DataFrame(preprocessor.fit_transform(data, y))
n_observation_X = 24 * 7*4  # For example, a week of data for the sequence
n_observation_y = 24 # We would like to forecast the 24 prices of the next day during the auction of today
n_sequence_train = 200
X_new, y_new = sequence_data_predict(X_preprocessed, y,
                  n_observation_X, n_observation_y,
                  n_sequence_train)

# Load model
model = load_model_cache()
# print(model.summary())

# Retrieve prediction
y_predict = model.predict(X_new)

# Historical prediction
df['price_day_ahead_predicted'] = df['price_day_ahead'].shift(24)

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox('Select country',
        ['ES', 'FR 🚧', 'DE 🚧', 'UK 🚧']
    )
with col2:
    currency = st.selectbox('Select price currency',
        ['EUR', 'USD', 'GBP']
    )

df['price_day_ahead'] = RATE[currency] * df['price_day_ahead']

# Table 1
'''
### Table 1
'''
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
lweek = today - timedelta(7)
table_yday = df[df['time'].dt.date == yday][['time', 'price_day_ahead']].drop_duplicates('time')
table_today = df[df['time'].dt.date == today][['time', 'price_day_ahead']].drop_duplicates('time')
table_lweek = df[df['time'].dt.date == lweek][['time', 'price_day_ahead']].drop_duplicates('time')
table_yday['hour'] = table_yday['time'].dt.hour.astype(str)
table_today['hour'] = table_today['time'].dt.hour.astype(str)
table_lweek['hour'] = table_lweek['time'].dt.hour.astype(str)
table_yday['price_day_ahead_yday'] = table_yday['price_day_ahead']
table_today['price_day_ahead_today'] = table_today['price_day_ahead']
table_lweek['price_day_ahead_lweek'] = table_lweek['price_day_ahead']
table_yday.set_index('hour', inplace=True)
table_today.set_index('hour', inplace=True)
table_lweek.set_index('hour', inplace=True)
table_diff_day = table_today[['price_day_ahead_today']].join(table_yday[['price_day_ahead_yday']], on='hour')
table_diff_week = table_today[['price_day_ahead_today']].join(table_lweek[['price_day_ahead_lweek']], on='hour')
table_today['change vs. today (%)'] = (table_diff_day['price_day_ahead_today']/table_diff_day['price_day_ahead_yday']-1)*100
table_today['change vs. last week (%)'] = (table_diff_week['price_day_ahead_today']/table_diff_week['price_day_ahead_lweek']-1)*100
table_today['change vs. today (%)'] = table_today['change vs. today (%)'].apply(lambda x: f'▲ {round(x,2)}' if x > 0 else f'▼ {round(x,2)}')
table_today['change vs. last week (%)'] = table_today['change vs. last week (%)'].apply(lambda x: f'▲ {round(x,2)}' if x > 0 else f'▼ {round(x,2)}')
table_today[f'predicted price ({currency})'] = table_today['price_day_ahead'].apply(lambda x: f'{round(x, 2)}')
table_today = table_today.reset_index()[['hour', f'predicted price ({currency})', 'change vs. today (%)', 'change vs. last week (%)']]
col1, col2 = st.columns(2)
with col1:
    st.table(table_today.iloc[0:-12].style.applymap(color_template, subset=['change vs. today (%)', 'change vs. last week (%)']))
with col2:
    st.table(table_today.iloc[12:24].style.applymap(color_template, subset=['change vs. today (%)', 'change vs. last week (%)']))

# Ticker
# For below text boxes
col1, col2, col3, col4 = st.columns(4)
with col1:
    current = df['price_day_ahead'].iloc[-1]
    st.metric(f'Current Eletricity Price ({currency})', round(current,1), delta = '-5.0 %') #change with formula
with col2:
    tomorrow = y_predict[-1].mean() * RATE[currency]
    st.metric('Tomorrow\'s Price Prediction', round(tomorrow,1), delta = f'{round((tomorrow/current-1)*100, 1)} %')
with col3:
    tomorrow_lowest = y_predict[-1].min() * RATE[currency]
    st.metric('Lowest Price Prediction (24 hr)', round(tomorrow_lowest,1), f'{round((tomorrow_lowest/current-1)*100, 1)} %')
with col4:
    tomorrow_highest = y_predict[-1].max() * RATE[currency]
    st.metric('Highest Price Prediction (24 hr)', round(tomorrow_highest,1), f'{round((tomorrow_highest/current-1)*100, 1)} %')

# Plot 1
'''
#
#
### Plot 1
'''
df_plot1 = df
col1, col2 = st.columns(2)
with col1:
    selected_plot1 = st.selectbox('Select hour',
        [f'{i} am' for i in range(0,12)] + [f'{i} pm' for i in range(12,24)]
    )
with col2:
    selected_rolling_window = int(st.radio('Select day window',
        [14,30,60],
        horizontal=True,
    ))
# Filter based on selected hour
hour_plot1 = int(selected_plot1.split(' ')[0])
df_plot1 = df_plot1[df_plot1['time'].dt.hour == hour_plot1]

df_plot1['price_day_ahead'] = RATE[currency] * df_plot1['price_day_ahead']
df_plot1[f'min_{selected_rolling_window}D'] = df_plot1['price_day_ahead'].rolling(selected_rolling_window).min()
df_plot1[f'mean_{selected_rolling_window}D'] = df_plot1['price_day_ahead'].rolling(selected_rolling_window).mean()
df_plot1[f'max_{selected_rolling_window}D'] = df_plot1['price_day_ahead'].rolling(selected_rolling_window).max()
# fig_plot1 = px.line(df_plot1,
#         x='time',
#         y=[f'min_{selected_rolling_window}D', f'mean_{selected_rolling_window}D', f'max_{selected_rolling_window}D'],
#         width=15,
#     )
fig_plot1 = create_plot_hourly(df_plot1,
    colors = ['purple', 'green', 'blue'],
    columns = [f'min_{selected_rolling_window}D', f'mean_{selected_rolling_window}D', f'max_{selected_rolling_window}D'],
    width = 1.5,
)

st.plotly_chart(fig_plot1, theme="streamlit", use_container_width=True)

# Plot 2
'''
### Plot 2
'''
df['30D_SMA'] = df['price_day_ahead'].rolling(24*30).mean()
# This is for streamlit main page (chart)
# Properties of figure
fig = create_plot_historical_data(df, colors=['white', 'green', 'cyan'], is_sma=True)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Plot 3
'''
## Plot 3
'''
df_hour = df
col1, col2 = st.columns(2)
with col1:
    selected_hour = st.selectbox('Select hour ',
        [f'{i} am' for i in range(0,12)] + [f'{i} pm' for i in range(12,24)]
    )
# Filter based on selected hour
if selected_hour != 'all':
    hour_value = int(selected_hour.split(' ')[0])
    df_hour = df_hour[df_hour['time'].dt.hour == hour_value]

df_hour['price_day_ahead'] = RATE[currency] * df_hour['price_day_ahead']

fig_hour = create_plot_historical_data(df_hour,
                                       colors=['rgba(200, 200, 200, 0.1)','rgba(255, 255, 255, 1)'],
                                       is_sma=False)

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

# Filter based on selected hour
# if selected_hour != 'all':
#     hour_value = int(selected_hour.split(' ')[0])
#     df = df[df['time'].dt.hour == hour_value]

# df['price_day_ahead'] = RATE[currency] * df['price_day_ahead']
# if selected_hour == 'all':
#     df['30D_SMA'] = df['price_day_ahead'].rolling(24*30).mean()
# else:
#     df['30D_SMA'] = df['price_day_ahead'].rolling(30).mean()


# st.markdown('# Electricity Price Prediction :zap:')

# #st.sidebar.markdown("# Input Parameters")


# currency = st.sidebar.selectbox('Select price currency',
#     ['European Euro', 'US Dollar', 'UK Poundsterling', 'Swiss Franc',
#      'Norwegian Krone', 'Denmark Krone']
# )

# selected_hour = st.sidebar.selectbox('Select hour',
#     ['all'] + [f'{i} am' for i in range(0,12)] + [f'{i} pm' for i in range(12,24)]
# )


# # This is for streamlit main page (chart)
# # Properties of figure
# xaxis=dict(
#     title=None,
#     showgrid=False,
#     showline=False,
#     zeroline=False,
#     rangeselector=dict(
#         buttons=list([
#             dict(count=24, label="24 hour", step="hour", stepmode="backward"),
#             dict(count=3, label="3 days", step="day", stepmode="backward"),
#             dict(count=7, label="1 week", step="day", stepmode="backward"),
#             dict(count=1, label="1 month", step="month", stepmode="backward"),
#             dict(count=1, label="1 year", step="year", stepmode="backward"),
#             dict(step="all"),
#         ])
#     )
# )
# yaxis=dict(
#     title=f'price day ahead per MWh (in {currency})',
#     showgrid=True,
#     showline=False,
#     zeroline=False,
# )
# legend=dict(
#     yanchor="bottom",
#     y=-0.3,
#     xanchor="left",
#     x=0,
#     title=None,
#     orientation='h',
# )

# fig = px.line(df,
#     x='time',
#     y=['price_day_ahead', 'price_day_ahead_predicted', '30D_SMA'],
#     width=15,
# )

# fig.update_layout(
#     xaxis=xaxis,
#     yaxis=yaxis,
#     legend=legend,
#     paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)',
#     template='plotly_dark',
#     xaxis_rangeselector_font_color='black',
#     xaxis_rangeselector_activecolor='gray',
#     xaxis_rangeselector_bgcolor='silver',
# )

# # Update legend name
# newnames = {
#     'price_day_ahead': 'Real price',
#     'price_day_ahead_predicted': 'Predicted price',
#     '30D_SMA': 'Real price 30D_SMA'
# }
# colors = {
#     'price_day_ahead': 'white',
#     'price_day_ahead_predicted': 'green',
#     '30D_SMA': 'cyan',
# }
# line_width = {
#     'price_day_ahead': 0.5,
#     'price_day_ahead_predicted': 1,
#     '30D_SMA': 2,
# }
# fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
#                                       legendgroup = newnames[t.name],
#                                       hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name]),
#                                       line = dict(color=colors[t.name], width=line_width[t.name]),
#                                      )
#                   )

# st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# # For below text boxes
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     current = df['price_day_ahead'].iloc[-1]
#     st.metric(f'Current Eletricity Price in {currency}', round(current,1), delta = '-5.0 %') #change with formula
# with col2:
#     tomorrow = y_predict[-1].mean() * RATE[currency]
#     st.metric('Tomorrow\'s Price Prediction', round(tomorrow,1), delta = f'{round((tomorrow/current-1)*100, 1)} %')
# with col3:
#     tomorrow_lowest = y_predict[-1].min() * RATE[currency]
#     st.metric('Lowest Price Prediction (24 hr)', round(tomorrow_lowest,1), f'{round((tomorrow_lowest/current-1)*100, 1)} %')
# with col4:
#     tomorrow_highest = y_predict[-1].max() * RATE[currency]
#     st.metric('Highest Price Prediction (24 hr)', round(tomorrow_highest,1), f'{round((tomorrow_highest/current-1)*100, 1)} %')


# st.sidebar.text("")
# st.sidebar.markdown("***")
# st.sidebar.text("")

# st.sidebar.markdown(
# '''
# ### :mailbox: Subscribe to our mailing list to receive updates and daily price information!
# '''
# )


# contact_form = """
# <form action="https://formsubmit.co/YOUREMAIL@EMAIL.COM" method="POST">
#      <input type="hidden" name="_captcha" value="false">
#      <input type="text" name="name" placeholder="Your name" required>
#      <input type="email" name="email" placeholder="Your email" required>
#      <button type="submit">Send</button>
# </form>
# """

# st.sidebar.markdown(contact_form, unsafe_allow_html=True)

# st.text("")
# st.markdown("***")
# st.text("")

# '''
# ##### Disclaimer
# "The information provided on this website is for educational and informational purposes only.
# It is not intended to be and does not constitute financial advice.
# NeuralEnergy does not make any representations or warranties about the accuracy or
# completeness of the information provided on this website.
# Any reliance you place on the information is strictly at your own risk.
# NeuralEnergy will not be liable for any losses or damages in connection
# with the use of this website or the information contained herein."
# '''

# # '''
# # #
# # #
# # #
# # '''
# # '''
# # ### Simple moving average over historical day ahead price
# # '''

# # df = df.set_index('time')
# # df_price = df[['price_day_ahead']]
# # df_price['hour'] = df_price.index.hour+1
# # df_price['7D_SMA'] = df_price['price_day_ahead'].rolling(24*7).mean()
# # df_price['30D_SMA'] = df_price['price_day_ahead'].rolling(24*30).mean()
# # df_price.dropna(inplace=True)

# # dict_of_hourly_df = {}
# # for i in range(24):
# #     key_name = i+1
# #     df_new = df_price[['price_day_ahead']][df_price['hour'] == i+1]
# #     df_new['30D_SMA'] = df_new['price_day_ahead'].rolling(30).mean()
# #     df_new['14D_SMA'] = df_new['price_day_ahead'].rolling(14).mean()
# #     df_new['7D_SMA'] = df_new['price_day_ahead'].rolling(7).mean()
# #     df_new['3D_SMA'] = df_new['price_day_ahead'].rolling(3).mean()
# #     df_new.dropna(inplace=True)
# #     dict_of_hourly_df[key_name] = df_new


# # fig2 = px.line(df_price, x=df_price.index, y=['price_day_ahead', '30D_SMA'], color_discrete_sequence=['rgba(200, 200, 200, 0.1)',
# #                                                                                                     'rgba(255, 255, 255, 1)'])


# # fig2.update_layout(legend=dict(
# #     yanchor="top",
# #     y=0.99,
# #     xanchor="left",
# #     x=0.01
# # ))

# # newnames2 = {'30D_SMA':'30D SMA', 'price_day_ahead': 'Hourly Price'}
# # fig2.for_each_trace(lambda t: t.update(name = newnames2[t.name],
# #                                     legendgroup = newnames2[t.name],
# #                                     hovertemplate = t.hovertemplate.replace(t.name, newnames2[t.name])
# #                                     )
# #                 )

# # fig2.update_layout(
# #     yaxis_title=f"Price (in {currency})",
# #     legend_title=None,
# #     font=dict(
# #         family="Arial",
# #         size=12,
# #         color="white"
# #     )
# # )

# # xaxis2=dict(
# #     title=None,
# #     showgrid=False,
# #     showline=False,
# #     zeroline=False,

# #     rangeselector=dict(
# #         buttons=list([
# #             dict(count=1, label="1 month", step="month", stepmode="backward"),
# #             dict(count=3, label="3 month", step="month", stepmode="backward"),
# #             dict(count=6, label="6 month", step="month", stepmode="backward"),
# #             dict(count=1, label="Year To Date", step="year", stepmode="todate"),
# #             dict(count=1, label="1 year", step="year", stepmode="backward"),
# #             dict(step="all")
# #         ])
# #     )
# # )

# # fig2.update_layout(
# # xaxis=xaxis2,
# # paper_bgcolor='rgba(0,0,0,0)',
# # plot_bgcolor='rgba(0,0,0,0)',
# # template='plotly_dark',
# # xaxis_rangeselector_font_color='black',
# # xaxis_rangeselector_activecolor='gray',
# # xaxis_rangeselector_bgcolor='silver'
# # )

# # st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
