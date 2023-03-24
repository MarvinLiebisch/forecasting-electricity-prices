import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
def predict():
    """
    Make a single course prediction.
    """
    # create pd dataframe from parameters

    # preprocess pd dataframe

    # predict dgn model
    predict = 20

    return {
        'price_day_ahead': predict,
    }


@app.get("/")
def root():
    return {'greeting': 'Haii...can not predict electricity price right now. Our team are working to finish the model first.Cheeeerss'}
