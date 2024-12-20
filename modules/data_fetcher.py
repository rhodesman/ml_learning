import yfinance as yf
import ccxt
import pandas as pd
from datetime import datetime, timedelta

def map_granularity(asset_type, granularity):
    """
    Maps the unified granularity to the specific format required by each data source.

    Args:
        asset_type (str): The type of asset ('stock' or 'crypto').
        granularity (str): Unified granularity format ('daily', 'hourly', etc.).

    Returns:
        str: Mapped granularity for the asset's data source.
    """
    if asset_type == 'stock':
        return {
            'daily': '1d',
            'hourly': '1h'
        }.get(granularity, '1d')  # Default to '1d' if granularity not mapped
    elif asset_type == 'crypto':
        return {
            'daily': 'daily',
            'hourly': 'hourly'
        }.get(granularity, 'daily')  # Default to 'daily' if granularity not mapped
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")

def fetch_stock_data(ticker, lookback_days, granularity):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        # Map granularity for stocks
        granularity = map_granularity('stock', granularity)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Download stock data
        df = yf.download(ticker, start=start_date, end=end_date, interval=granularity)

        if df.empty:
            print(f"Warning: No data returned for stock {ticker}")
            return None

        # Format and return the data
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'time', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
        return df[['time', 'adj_close', 'volume']]

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def fetch_crypto_data(crypto_name, lookback_days, granularity):
    """
    Fetch historical cryptocurrency data.
    """
    try:
        # Map granularity for cryptocurrencies
        granularity = map_granularity('crypto', granularity)

        # Simulating crypto data fetch (replace with your API call)
        # Example: Replace the following lines with the actual crypto API logic
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        df = simulate_crypto_api(crypto_name, start_date, end_date, granularity)

        if df.empty:
            print(f"Warning: No data returned for crypto {crypto_name}")
            return None

        df.reset_index(inplace=True)
        df.rename(columns={'timestamp': 'time', 'price': 'price', 'volume': 'volume'}, inplace=True)
        return df[['time', 'price', 'volume']]

    except Exception as e:
        print(f"Error fetching crypto data for {crypto_name}: {e}")
        return None