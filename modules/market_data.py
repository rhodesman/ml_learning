import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()

def fetch_crypto_data_coingecko(crypto_id, days):
    """
    Fetch historical cryptocurrency data from CoinGecko.

    Args:
        crypto_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin").
        days (int): Number of days to look back.

    Returns:
        pd.DataFrame: DataFrame containing historical crypto data.
    """
    print(f"Fetching {days} days of data for {crypto_id}...")
    data = cg.get_coin_market_chart_by_id(id=crypto_id, vs_currency="usd", days=days)
    prices = pd.DataFrame(data["prices"], columns=["time", "price"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["time", "volume"])
    df = pd.merge(prices, volumes, on="time")
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df


def fetch_stock_data(ticker="AAPL", days=30):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "time", "Adj Close": "adj_close", "Volume": "volume"}, inplace=True)
    return df


def fetch_news_data(query="cryptocurrency", days=29):
    """
    Fetch news articles using NewsAPI.

    Args:
        query (str): Search query for news articles.
        days (int): Number of days to fetch articles for.

    Returns:
        pd.DataFrame: A DataFrame with news data.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("Missing NewsAPI key. Please add it to your .env file.")

    # Define the date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    print(f"Requesting news from {start_date.isoformat()} to {end_date.isoformat()}")  # Debug

    # NewsAPI endpoint and parameters
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "sortBy": "publishedAt",
        "pageSize": 100,  # Maximum articles per request
        "apiKey": api_key,
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Error fetching news data: {response.status_code} - {response.text}")

    articles = response.json().get("articles", [])
    data = [
        {
            "publishedAt": article["publishedAt"],
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "source": article["source"]["name"],
        }
        for article in articles
    ]

    return pd.DataFrame(data)


def save_to_csv(dataframe, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dataframe.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def collect_crypto_data(crypto_ids, days):
    """
    Collect and save cryptocurrency data for multiple cryptocurrencies.

    Args:
        crypto_ids (list): List of CoinGecko cryptocurrency IDs.
        days (int): Number of days to look back.
    """
    for crypto in crypto_ids:
        df = fetch_crypto_data_coingecko(crypto, days)
        filename = f"data/raw/{crypto}_data.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


def collect_stock_data():
    tickers = ["AAPL", "TSLA", "MSFT"]
    for ticker in tickers:
        df = fetch_stock_data(ticker=ticker, days=30)
        save_to_csv(df, f"data/raw/{ticker}_stock_data.csv")


def collect_news_data():
    queries = ["cryptocurrency", "stock market", "technology"]
    for query in queries:
        df_news = fetch_news_data(query=config["news_query"], days=config["lookback_days"])
        save_to_csv(df_news, "data/raw/news_data.csv")