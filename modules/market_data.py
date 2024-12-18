import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables
load_dotenv()

# API credentials from .env
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")


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

    # Debugging: Print the constructed URL and headers
    print("Constructed URL:", url)
    print("Headers:", headers)

    # Make the API request
    response = requests.get(url, headers=headers)

    # Print the response for debugging
    print("Response Code:", response.status_code)
    print("Response Body:", response.text)

    # Raise an error for non-200 responses
    response.raise_for_status()

    # Parse response JSON into a DataFrame
    data = response.json().get("candles", [])
    if not data:
        raise ValueError("No candlestick data found in the API response.")

    # Format the data into a DataFrame
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    return df


def fetch_stock_data(symbol="AAPL", days=30):
    """
    Placeholder function for fetching stock data.

    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL").
        days (int): Number of days to fetch historical data for.

    Returns:
        pd.DataFrame: A DataFrame with stock data.
    """
    print(f"Fetching stock data for {symbol} (last {days} days)...")
    # You can add logic to fetch stock data using yfinance or similar libraries
    return pd.DataFrame()


def fetch_news_data(query="cryptocurrency", days=30):
    """
    Placeholder function for fetching news data.

    Args:
        query (str): Search query for news articles.
        days (int): Number of days to fetch historical data for.

    Returns:
        pd.DataFrame: A DataFrame with news data.
    """
    print(f"Fetching news data for query: {query} (last {days} days)...")
    # You can add logic to fetch news data from a news API
    return pd.DataFrame()