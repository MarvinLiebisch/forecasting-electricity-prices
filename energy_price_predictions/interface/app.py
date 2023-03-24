import streamlit as st

'''
# Energy price prediction frontend
'''

st.markdown('Front-end examples')

# from datetime import datetime

# pickup_date = st.date_input('Enter pickup date', value=datetime.now())
# pickup_time = st.time_input('Enter pickup time', value=datetime.now().time())
# pickup_longitude = float(st.text_input('Enter pickup longitude', value='-73.950655'))
# pickup_latitude = float(st.text_input('Enter pickup longitude', value='40.783282'))
# dropoff_longitude = float(st.text_input('Enter dropoff longitude', value='-73.984365'))
# dropoff_latitude = float(st.text_input('Enter dropoff longitude', value='40.769802'))
# passenger_count = int(st.slider('Number of passengers', min_value=0, max_value=20, value=1))


# # additional map
# import pandas as pd

# location = pd.DataFrame({
#     'longitude': [pickup_longitude, dropoff_longitude],
#     'latitude': [pickup_latitude, dropoff_latitude],
# })
# st.map(location)

# # '''
# # ## Once we have these, let's call our API in order to retrieve a prediction

# # See ? No need to load a `model.joblib` file in this app, we do not even need to know anything about Data Science in order to retrieve a prediction...

# # 🤔 How could we call our API ? Off course... The `requests` package 💡
# # '''

# url = 'https://taxifare.lewagon.ai/predict'

# # if url == 'https://taxifare.lewagon.ai/predict':

# #     st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')

# # '''

# # 2. Let's build a dictionary containing the parameters for our API...
# # https://taxifare.lewagon.ai/predict?pickup_datetime=2014-07-06%2019:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2

# request_body = {
#     'pickup_datetime': f'{pickup_date.isoformat()} {pickup_time.isoformat()}',
#     'pickup_longitude': pickup_longitude,
#     'pickup_latitude': pickup_latitude,
#     'dropoff_longitude': dropoff_longitude,
#     'dropoff_latitude': dropoff_latitude,
#     'passenger_count': passenger_count,
# }
# print(request_body)
# # 3. Let's call our API using the `requests` package...

# # if submit_button:
# import requests
# response = requests.get(url, params=request_body)
# print(response.text)

# # 4. Let's retrieve the prediction from the **JSON** returned by the API...
# import json
# prediction = json.loads(response.text).get('fare')

# # ## Finally, we can display the prediction to the user
# # '''
# st.markdown(f'Estimated fare of your journey is ${round(prediction,2)}')
