import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, lookback_days, granularity):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start_date, end=end_date, interval=granularity)
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'time', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
    return df

def fetch_crypto_data(crypto_name, lookback_days, granularity):
    # Implement the function to fetch cryptocurrency data
    pass