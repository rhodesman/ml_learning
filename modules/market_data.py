import yfinance as yf
from datetime import datetime, timedelta, timezone
import os
import hmac
import hashlib
import time
from dotenv import load_dotenv
import requests
import pandas as pd
from base64 import b64encode
from coinbase.rest import RESTClient
import logging

logging.basicConfig(level=logging.DEBUG) 

# Load environment variables
load_dotenv()

# API credentials from .env
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")

# Initialize the RESTClient
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

def fetch_crypto_data(product_id="BTC-USD", days=30, granularity=86400):
    """
    Fetch historical cryptocurrency candlestick data from Coinbase API.

    Args:
        product_id (str): The product ID (e.g., "BTC-USD").
        days (int): Number of days to fetch historical data for.
        granularity (int): Granularity in seconds (e.g., 86400 for 1 day).

    Returns:
        pd.DataFrame: A DataFrame with time, low, high, open, close, and volume.
    """
    # Define the time range with timezone information
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=days)

    # Format timestamps as ISO 8601 strings with "Z" suffix
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Debug timestamps
    print("Start Time (string):", start_time_str)
    print("End Time (string):", end_time_str)

    # Fetch candles using the RESTClient
    candles = client.get_candles(
        product_id=product_id,
        start=start_time_str,
        end=end_time_str,
        granularity=granularity  # Ensure this is an integer
    )

    # Parse the response into a DataFrame
    data = [
        {
            "time": candle.start,
            "low": candle.low,
            "high": candle.high,
            "open": candle.open,
            "close": candle.close,
            "volume": candle.volume
        }
        for candle in candles.candles
    ]

    return pd.DataFrame(data)

def fetch_crypto_data_manual(product_id="BTC-USD", days=30, granularity=86400):
    # Define the time range with timezone information
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=days)

    # Format timestamps as ISO 8601 strings with "Z" suffix
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Manually construct the URL query string
    url = (
        f"https://api.coinbase.com/api/v3/brokerage/products/{product_id}/candles"
        f"?start={start_time_str}&end={end_time_str}&granularity={granularity}"
    )

    # Construct headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Print the constructed URL for debugging
    print("Constructed URL:", url)

    # Make the API request
    response = requests.get(url, headers=headers)

    # Print the response for debugging
    print("Response Code:", response.status_code)
    print("Response Body:", response.text)

    # Raise an error for non-200 responses
    response.raise_for_status()

    # Parse response JSON
    return response.json()

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data
