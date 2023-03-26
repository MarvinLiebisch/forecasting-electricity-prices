import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.express as px

from energy_price_predictions.ml_logic.data_import import *
from energy_price_predictions.ml_logic.visualization import *



'''
# Energy price predictions
'''

st.sidebar.markdown("# Input Parameters")

#-- Set time by GPS or event
start_date = st.sidebar.date_input('Start date', value=date(2019,1,1))
start_time = datetime.strptime(f'{str(start_date)} 00:00:00+01:00', '%Y-%m-%d %H:%M:%S%z')
prediction = st.sidebar.selectbox('Prediction Interval',
                                    ['Hourly', '3-day', '7-day'])
print(start_date)
print(prediction)


df = import_merged_data()

if prediction == 'Hourly':
    start_past = start_date - timedelta(hours=48)
    past_df = df.loc[df.index >= str(start_past)][['price_day_ahead']].reset_index()
    past_df['type'] = 'historical price'
    future_df = pd.DataFrame({
        'time': np.array([start_time + timedelta(hours=i) for i in range(24)]),
        'price_day_ahead': [64.3 + np.random.random()*6 for i in range(24)] # derived from API
    })
    future_df['type'] = 'prediction price'
    merged_df = pd.concat([past_df, future_df])

    fig = px.line(merged_df, x='time', y='price_day_ahead', color='type')
    st.plotly_chart(fig, theme="streamlit")
