import streamlit as st
import pandas as pd
import numpy as np

from energy_price_predictions.ml_logic.data_import import *
from energy_price_predictions.ml_logic.visualization import *



'''
# Energy price predictions
'''

st.markdown('Visualization')

df = import_merged_data()

fig = plt.figure(figsize=(15,12.5))

# .corr heatmap of df to visualize correlation & show plot
sns.heatmap(round(df.corr(),1), annot=False, cmap='Purples', linewidth=0.9)

st.pyplot(fig)
