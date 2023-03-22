# **Forecasting Electricity Prices**

## Project

### Background

<ins>Day-ahead auctions</ins>

Day-ahead auctions are used in electricity markets to facilitate the purchase and sale of electricity for the following day. These auctions provide market participants with an opportunity to bid on the amount of electricity they require or can supply, at a specified price. The clearing price of the auction is determined by the intersection of the bids and offers, and sets the price for all market participants who buy or sell electricity in the day-ahead market for the following day.


### Model

Our model is predicting the price of the aforementioned day-ahead auctions. We do so by using time-series data for power generation, weather and previously observed prices in both the day-ahead market and spot market.


## Setup

### Data

This project uses the [Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) from Kaggle. 

### Content
This dataset contains 4 years of electrical consumption, generation, pricing, and weather data for Spain (2015-2018). Consumption and generation data was retrieved from ENTSOE a public portal for Transmission Service Operator (TSO) data. Settlement prices were obtained from the Spanish TSO Red Electric España. The weather data originates from the Open Weather API for the 5 largest cities in Spain.


### Installation

Python version 3.10.6

*Add tree here*

Required libraries can be installed from the *forecasting-electricity-data* directory by running:
```
pip install -r requirements.txt
```
