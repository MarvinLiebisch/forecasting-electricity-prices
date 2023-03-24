FROM python:3.10.6-slim-buster

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

# Then only, install application
COPY energy_price_predictions /energy_price_predictions
COPY models /models
COPY setup.py /setup.py
RUN pip install -e .

CMD uvicorn energy_price_predictions.api.fast:app --reload --host 0.0.0.0 --port $PORT
