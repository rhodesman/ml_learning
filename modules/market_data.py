import yfinance as yf
from datetime import datetime, timedelta, timezone
import os
import requests
import pandas as pd
from base64 import b64encode
import logging

#logging.basicConfig(level=logging.DEBUG) 

def fetch_crypto_data(coin_id="bitcoin", vs_currency="usd", days=30):
    """
    Fetch historical cryptocurrency data using CoinGecko API.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin").
        vs_currency (str): The fiat currency to compare against (e.g., "usd").
        days (int): Number of days to fetch historical data for.

    Returns:
        pd.DataFrame: A DataFrame with time, price, and volume data.
    """
    # Construct the API URL
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily"  # Options: "minutely", "hourly", "daily"
    }

    # Make the API request
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Error fetching data from CoinGecko API: {response.status_code} - {response.text}")

    # Parse the response data
    data = response.json()
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])

    # Format the data into a DataFrame
    df_prices = pd.DataFrame(prices, columns=["time", "price"])
    df_volumes = pd.DataFrame(volumes, columns=["time", "volume"])
    df = pd.merge(df_prices, df_volumes, on="time")

    # Convert timestamp to datetime
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data
