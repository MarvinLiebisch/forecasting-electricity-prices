# **Forecasting Electricity Prices**

## Project objectives

### Background

<ins>Day-ahead auctions</ins>

Day-ahead auctions are used in electricity markets to facilitate the purchase and sale of electricity for the following day. These auctions provide market participants with an opportunity to bid on the amount of electricity they require or can supply, at a specified price. The clearing price of the auction is determined by the intersection of the bids and offers, and sets the price for all market participants who buy or sell electricity in the day-ahead market for the following day.

### Model

Our model is predicting the price of the aforementioned day-ahead auctions, in order to find a better estimate of the amount one should bid.


## Setup

### Data

This project uses the [Hourly energy demand generation and weather](https://www.notion.so/%5B%3Chttps://link-url-here.org%3E%5D(%3Chttps://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather%3E)) from Kaggle. It contains energy

###Content
This dataset contains 4 years of electrical consumption, generation, pricing, and weather data for Spain. Consumption and generation data was retrieved from ENTSOE a public portal for Transmission Service Operator (TSO) data. Settlement prices were obtained from the Spanish TSO Red Electric España. Weather data was purchased as part of a personal project from the Open Weather API for the 5 largest cities in Spain and made public here.


### Installation

Python version 3.10.6

*Add tree here*

Required libraries can be installed from the *forecasting_electricity-data* directory by running:
```
pip install -r requirements.txt
```
