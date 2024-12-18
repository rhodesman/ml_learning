import yfinance as yf
from datetime import datetime, timedelta, timezone
import os
import hmac
import hashlib
import time
import jwt
from dotenv import load_dotenv
import requests
import pandas as pd
from base64 import b64encode

# Load environment variables
load_dotenv()

# API credentials from .env
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")

def generate_jwt(api_key, api_secret):
    """
    Generate a JWT for Coinbase API authentication.

    Args:
        api_key (str): The API key.
        api_secret (str): The API secret.

    Returns:
        str: The generated JWT.
    """
    # Current UNIX timestamp
    now = int(time.time())

    # JWT payload
    payload = {
        "iat": now,  # Issued at time
        "exp": now + 300,  # Expiration time (5 minutes from now)
    }

    # Generate the JWT
    token = jwt.encode(payload, api_secret, algorithm="HS256", headers={"kid": api_key})
    return token

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
    # Generate the JWT
    jwt_token = generate_jwt(API_KEY, API_SECRET)

    # Define the time range with timezone information
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=days)

    # API endpoint and request path
    base_url = "https://api.coinbase.com/api/v3/brokerage"
    request_path = f"/products/{product_id}/candles"
    url = f"{base_url}{request_path}?start={start_time.isoformat()}&end={end_time.isoformat()}&limit={days}&granularity={granularity}"

    # Headers with the JWT
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }

    # Debugging request
    print("Request URL:", url)
    print("Headers:", headers)

    # Make the API request
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data from Coinbase API: {response.status_code} - {response.text}")

    # Parse response JSON into a DataFrame
    data = response.json().get("candles", [])
    if not data:
        raise ValueError("No candlestick data found in the API response.")

    return pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data
