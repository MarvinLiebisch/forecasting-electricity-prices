import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.express as px
import requests
import json

from energy_price_predictions.ml_logic.data_import import *
from energy_price_predictions.ml_logic.visualization import *
from energy_price_predictions.ml_logic.model import *
from energy_price_predictions.ml_logic.registry import *
from energy_price_predictions.ml_logic.preprocessing import *

#API_URL = "https://electricitypricepredictions-nrgmmiocwa-ew.a.run.app/predict"
RATE = {
    'European Euro': 1,
    'US Dollar': 1.08,
    'UK Poundsterling': 0.88,
    'Swiss Franc': 0.99,
    'Denmark Krone': 7.46,
    'Norwegian Krone': 11.28,
}

st.set_page_config(page_title="Electricy Price Prediction", layout="wide") #page_icon=img,

st.markdown('# Electricity Price Prediction')

#st.sidebar.markdown("# Input Parameters")


currency = st.sidebar.selectbox('Select price currency',
    ['European Euro', 'US Dollar', 'UK Poundsterling', 'Swiss Franc',
     'Norwegian Krone', 'Denmark Krone']
)

selected_hour = st.sidebar.selectbox('Select hour',
    ['all'] + [f'{i} am' for i in range(0,12)] + [f'{i} pm' for i in range(12,24)]
)


# Retrieve data
df = import_merged_data_cache().reset_index()

# Filter based on selected hour
if selected_hour != 'all':
    hour_value = int(selected_hour.split(' ')[0])
    df = df[df['time'].dt.hour == hour_value]

df['price_day_ahead'] = RATE[currency] * df['price_day_ahead']
# Load model
model = load_model_from_mlflow()
print(model.summary())

# Retrieve prediction
## Preprocessing data
# X_preprocessed = ...
# y = df['price_day_ahead']
## Predict
# y_predict = model.predict(X_preprocessed, y)
# df['price_day_ahead_predicted'] = y_predict

# dummy (deleted)
df['price_day_ahead_predicted'] = df['price_day_ahead'].shift(24)

# future_df = pd.DataFrame({
#     'time': np.array([df['time'].max() + timedelta(hours=i) for i in range(1,25)]),
#     'price_day_ahead_predicted': [50 + np.random.random()*20 for i in range(24)],
# })


# This is for streamlit main page (chart)
# Properties of figure
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
    y=['price_day_ahead', 'price_day_ahead_predicted'],
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
colors = {
    'price_day_ahead': 'white',
    'price_day_ahead_predicted': 'green',
}
line_width = {
    'price_day_ahead': 1,
    'price_day_ahead_predicted': 1.5,
}
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name]),
                                      line = dict(color=colors[t.name], width=line_width[t.name]),
                                     )
                  )

st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# For below text boxes
col1, col2, col3, col4 = st.columns(4)
with col1:
    current = df['price_day_ahead'].iloc[-1]
    st.metric(f'Current Eletricity Price in {currency}', round(current,1), delta = '-5.0 %') #change with formula
with col2:
    # tomorrow = df['price_day_ahead_predicted'].iloc[-1]
    # dummy
    tomorrow = 67.8 * RATE[currency]
    st.metric('Tomorrow\'s Price Prediction', round(tomorrow,1), delta = f'{round((tomorrow/current-1)*100, 1)} %')
with col3:
    # dummy
    tomorrow_lowest = 63.3 * RATE[currency]
    st.metric('Lowest Price Prediction (24 hr)', round(tomorrow_lowest,1), f'{round((tomorrow_lowest/current-1)*100, 1)} %')
with col4:
    # dummy
    tomorrow_highest = 70.1 * RATE[currency]
    st.metric('Highest Price Prediction (24 hr)', round(tomorrow_highest,1), f'{round((tomorrow_highest/current-1)*100, 1)} %')
