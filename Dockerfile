FROM python:3.10.6-slim-buster

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt
RUN apt update && apt install git -yq

# Then only, install application
COPY energy_price_predictions /energy_price_predictions
COPY models /models
COPY raw_data /raw_data
COPY setup.py /setup.py
RUN pip install -e .

#CMD uvicorn energy_price_predictions.api.fast:app --reload --host 0.0.0.0 --port $PORT
CMD streamlit run energy_price_predictions/interface/app.py --server.port 8080 --server.enableCORS=false
