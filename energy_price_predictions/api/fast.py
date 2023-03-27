import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import numpy as np

app = FastAPI()

# add model
# app.state.model = load_model()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?...
@app.get("/predict")
def predict(start_date, prediction_interval):
    """
    Make a single course prediction.
    """
    # create pd dataframe from parameters
    start_time = datetime.strptime(f'{str(start_date)} 00:00:00+01:00', '%Y-%m-%d %H:%M:%S%z')

    # preprocess pd dataframe

    # predict dgn model
    # data simulation
    if prediction_interval == 'Hourly':
        future_df = pd.DataFrame({
            'time': np.array([start_time + timedelta(hours=i) for i in range(24)]),
            'price_day_ahead': [50 + np.random.random()*20 for i in range(24)] # derived from API
        })
    return future_df.to_dict()


@app.get("/")
def root():
    return {'greeting': 'Haii...can not predict electricity price right now. Our team are working to finish the model first.Cheeeerss'}
