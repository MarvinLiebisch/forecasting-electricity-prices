import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

# import sys
# sys.path.insert(0,'/home/rahmah/code/marvinliebisch/forecasting-electricity-prices')

from energy_price_predictions.interface.main import *


PREFECT_FLOW_NAME = "electricitypriceprediction"


@task
def preprocess_new_data():
    return preprocess()

@task
def evaluate_production_model(X_new, y_new):
    return evaluate(X_new, y_new)

@task
def re_train(X_train, y_train, X_val, y_val):
    return train(X_train, y_train, X_val, y_val)

# @task
# def transition_model(current_stage: str, new_stage: str):
#     return mlflow_transition_model(current_stage=current_stage, new_stage=new_stage)

# @task
# def notify(old_mae, new_mae):
#     """
#     Notify about the performance
#     """
#     base_url = 'https://wagon-chat.herokuapp.com'
#     channel = '802'
#     url = f"{base_url}/{channel}/messages"
#     author = 'krokrob'
#     if new_mae < old_mae and new_mae < 2.5:
#         content = f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"
#     elif old_mae < 2.5:
#         content = f"✅ Old model still good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
#     else:
#         content = f"🚨 No model good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
#     data = dict(author=author, content=content)
#     response = requests.post(url, data=data)
#     response.raise_for_status()


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the prefect workflow for the `taxifare` package. It should:
    - preprocess 1 month of new data, starting from EVALUATION_START_DATE
    - compute `old_mae` by evaluating current production model in this new month period
    - compute `new_mae` by re-training then evaluating current production model on this new month period
    - if new better than old, replace current production model by new one
    - if neither models are good enough, send a notification!
    """

    preprocessed = preprocess_new_data.submit()
    X_train, X_val, X_test, y_train, y_val,y_test = preprocessed.result()
    # old_mae = evaluate_production_model.submit(min_date=min_date, max_date=max_date, wait_for=[preprocessed])
    new_mae = re_train.submit(X_train, X_val, y_train, y_val, wait_for=[preprocessed])
    # old_mae = old_mae.result()
    # new_mae = new_mae.result()
    # if new_mae < old_mae:
    #     print(f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}")
    #     transition_model.submit(current_stage="Staging", new_stage="Production")
    # else:
    #     print(f"🚀 Old model kept in place with MAE: {old_mae}. The new MAE was: {new_mae}")
    # notify.submit(old_mae, new_mae)

if __name__ == "__main__":
    train_flow()
