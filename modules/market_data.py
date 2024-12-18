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

# Load environment variables
load_dotenv()

# API credentials from .env
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")

def generate_signature(timestamp, method, request_path, body, secret):
    """
    Generate the CB-ACCESS-SIGN header for Coinbase API authentication.

    Args:
        timestamp (str): The current UNIX timestamp as a string.
        method (str): HTTP method (GET, POST, etc.).
        request_path (str): The API endpoint path.
        body (str): The request body (empty string for GET requests).
        secret (str): The API secret.

    Returns:
        str: The generated signature.
    """
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    signature = hmac.new(
        b64encode(secret.encode()),
        message.encode(),
        hashlib.sha256
    ).digest()
    return b64encode(signature).decode()

def fetch_crypto_data(product_id="BTC-USD", days=30, granularity="ONE_DAY"):
    """
    Fetch historical cryptocurrency data from Coinbase API.

    Args:
        product_id (str): The product ID (e.g., "BTC-USD").
        days (int): Number of days to fetch historical data for.
        granularity (str): Granularity of candlesticks (e.g., "ONE_DAY", "ONE_MINUTE").

    Returns:
        pd.DataFrame: A DataFrame with time, open, high, low, close, and volume.
    """
    # Ensure product_id is correctly formatted
    if not product_id.endswith("USD"):
        raise ValueError(f"Invalid product_id format: {product_id}. Expected format is BASE-QUOTE (e.g., BTC-USD).")

    # Define the time range with timezone information
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=days)

    # API endpoint and request path
    base_url = "https://api.coinbase.com/api/v3/brokerage"
    request_path = f"/products/{product_id}/candles"
    url = base_url + request_path

    # Query parameters
    params = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "granularity": granularity
    }

    # Generate authentication headers
    timestamp = str(int(time.time()))
    method = "GET"
    body = ""  # Empty body for GET requests
    signature = generate_signature(timestamp, method, request_path, body, API_SECRET)

    headers = {
        "CB-ACCESS-KEY": API_KEY,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-VERSION": "2021-03-23"  # API version date
    }

    # Make the API request
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data from Coinbase API: {response.status_code} - {response.text}")
        print("Request URL:", url)
        print("Headers:", headers)
        print("Params:", params)
        print("Response:", response.text)

    # Parse response JSON into a DataFrame
    data = response.json().get("candles", [])
    if not data:
        raise ValueError("No candlestick data found in the API response.")

    return pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data
