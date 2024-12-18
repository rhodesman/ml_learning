import yfinance as yf
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Get API credentials from the .env file
api_key = os.getenv("COINBASE_API_KEY")
api_secret = os.getenv("COINBASE_API_SECRET")

# Initialize the Coinbase API client
client = RESTClient(api_key=api_key, api_secret=api_secret)

from datetime import datetime, timedelta, timezone

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
    # Define the time range with timezone information
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    # Print time for debugging
    print(f"Start Time: {start_time.isoformat()}")
    print(f"End Time: {end_time.isoformat()}")

    # Fetch candlestick data
    candles = client.get_candles(
        product_id=product_id,
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        granularity=granularity
    )

    # Convert the data to a DataFrame
    data = [
        {
            "time": candle.start,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume
        }
        for candle in candles.candles
    ]

    return pd.DataFrame(data)

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data
