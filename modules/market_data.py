import os
import ta
import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables from .env file
load_dotenv()

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


def fetch_news_data(query, days):
    """
    Fetch news data using Bing News Search API.

    Args:
        query (str): The search query for news articles.
        days (int): Number of days to look back.

    Returns:
        pd.DataFrame: News articles with publication dates and titles.
    """
    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        raise ValueError("Missing Bing API key. Please add it to your .env file.")

    url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "freshness": "Week",  # Adjust freshness as needed: Day, Week, or Month
        "count": 100,         # Number of articles to return
        "sortBy": "Date"      # Sort by most recent
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"Error fetching news data: {response.status_code} - {response.text}")

    articles = response.json().get("value", [])
    data = [
        {
            "title": article["name"],
            "description": article.get("description", ""),
            "url": article["url"],
            "publishedAt": article["datePublished"]
        }
        for article in articles
    ]

    df = pd.DataFrame(data)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the dataset.

    Args:
        df (pd.DataFrame): Dataset containing price-related columns.

    Returns:
        pd.DataFrame: Dataset with additional technical indicators.
    """
    # Handle MultiIndex columns for stock data
    if isinstance(df.columns, pd.MultiIndex):
        print("Detected MultiIndex columns. Flattening...")
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Determine the price column
    price_column = None
    for col in ['price', 'Close_AAPL', 'adj_close_AAPL', 'Close', 'adj_close']:
        if col in df.columns:
            price_column = col
            break

    if not price_column:
        raise KeyError("No valid price column found in the dataset.")

    # Add technical indicators
    print(f"Using price column: {price_column}")
    df["rsi"] = ta.momentum.RSIIndicator(df[price_column]).rsi()
    df["macd"] = ta.trend.MACD(df[price_column]).macd()
    bollinger = ta.volatility.BollingerBands(df[price_column])
    df["bollinger_hband"] = bollinger.bollinger_hband()
    df["bollinger_lband"] = bollinger.bollinger_lband()

    return df

def add_sentiment_scores(news_df):
    """
    Add sentiment scores to news articles.

    Args:
        news_df (pd.DataFrame): News dataset with a 'description' column.

    Returns:
        pd.DataFrame: News dataset with a 'sentiment' column.
    """
    analyzer = SentimentIntensityAnalyzer()
    news_df["sentiment"] = news_df["description"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"] if pd.notna(x) else 0
    )
    return news_df

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
        print("Columns in DataFrame:", df.columns)
        df = add_technical_indicators(df)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            print("Flattened Columns:", df.columns)

        filename = f"data/raw/{crypto}_data.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


def collect_stock_data(stocks, days):
    """
    Collect and save stock data for multiple stocks.

    Args:
        stocks (list): List of stock tickers.
        days (int): Number of days to look back.
    """
    for stock in stocks:
        
        print(f"Fetching {days} days of data for {stock}...")
        df = fetch_stock_data(ticker=stock, days=days)
        print("Columns in DataFrame:", df.columns)
        df = add_technical_indicators(df)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            print("Flattened Columns:", df.columns)

        filename = f"data/raw/{stock}_stock_data.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


def collect_news_data(query, days):
    """
    Collect and save news data.

    Args:
        query (str): The search query for news articles.
        days (int): Number of days to look back.
    """
    print(f"Fetching news data for query: {query} over {days} days...")
    df_news = fetch_news_data(query=query, days=days)
    df_news = add_sentiment_scores(df_news)
    filename = "data/raw/news_data.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_news.to_csv(filename, index=False)
    print(f"Data saved to {filename}")