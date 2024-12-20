import yfinance as yf
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI

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
    Fetch historical cryptocurrency data using CoinGeckoAPI.
    
    Args:
        crypto_name (str): Name of the cryptocurrency (e.g., 'bitcoin').
        lookback_days (int): Number of days to look back for historical data.
        granularity (str): Data granularity ('daily' or 'hourly').

    Returns:
        pd.DataFrame: A DataFrame with time, price, and volume data.
    """
    cg = CoinGeckoAPI()
    try:
        # Map granularity for cryptocurrencies
        granularity = map_granularity('crypto', granularity)
        
        # Get the current time
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Convert to Unix timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        # Fetch market chart data
        data = cg.get_coin_market_chart_range_by_id(
            id=crypto_name,
            vs_currency="usd",
            from_timestamp=start_timestamp,
            to_timestamp=end_timestamp
        )

        # Extract the data based on granularity
        if granularity == "daily":
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

        # Combine the data into a DataFrame
        df = pd.DataFrame({
            "time": [datetime.fromtimestamp(p[0] / 1000) for p in prices],
            "price": [p[1] for p in prices],
            "volume": [v[1] for v in volumes]
        })

        if df.empty:
            print(f"Warning: No data returned for crypto {crypto_name}")
            return None

        return df

    except Exception as e:
        print(f"Error fetching crypto data for {crypto_name}: {e}")
        return None