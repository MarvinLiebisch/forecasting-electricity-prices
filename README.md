# **Forecasting Electricity Prices**

## Project

### Background

<ins>Day-ahead auctions</ins>

Day-ahead auctions are used in electricity markets to facilitate the purchase and sale of electricity for the following day. These auctions provide market participants with an opportunity to bid on the amount of electricity they require or can supply, at a specified price. The clearing price of the auction is determined by the intersection of the bids and offers, and sets the price for all market participants who buy or sell electricity in the day-ahead market for the following day.


### Model

Our LSTM model is predicting the price of the aforementioned day-ahead auctions. We do so by using time-series data for power generation and previously observed prices.

### Front End

The generated predictions as well as historical data are visualized here: [Streamlit](https://forecasting-electricity-prices.streamlit.app/)


## Setup

### Data

This project uses energy market data obtained from the [European Network of Transmission System Operators for Electricity](https://www.entsoe.eu/). 
The dataset contains electrical consumption, generation, and pricing data since 2019.


### Installation

Python version 3.10.6

Required libraries can be installed from the *forecasting-electricity-data* directory by running:
```
pip install -r requirements.txt
```
