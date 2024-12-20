import yfinance as yf
import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, lookback_days, granularity):
    """
    Fetch historical stock data using yfinance.

    Args:
        ticker (str): The stock ticker (e.g., 'AAPL').
        lookback_days (int): Number of days of historical data to fetch.
        granularity (str): Data granularity (e.g., '1d' for daily, '1h' for hourly).

    Returns:
        pd.DataFrame: Processed stock data with columns ['time', 'adj_close', 'volume'].
    """
    try:
        # Calculate the start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Download stock data
        df = yf.download(ticker, start=start_date, end=end_date, interval=granularity)

        # Check if the data is empty
        if df.empty:
            print(f"Warning: No data returned for stock {ticker}")
            return None

        # Reset the index and rename columns for consistency
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'time', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)

        # Return only the relevant columns
        return df[['time', 'adj_close', 'volume']]

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def fetch_crypto_data(ticker, lookback_days, granularity='daily'):
    """
    Fetch historical cryptocurrency data using the Binance exchange.

    Args:
        ticker (str): The cryptocurrency ticker (e.g., 'BTC/USDT').
        lookback_days (int): Number of days of historical data to fetch.
        granularity (str): Data granularity ('daily' or 'hourly').

    Returns:
        pd.DataFrame: Historical data with columns ['time', 'price', 'volume'].
    """
    try:
        exchange = ccxt.binance()  # Initialize the Binance exchange
        since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)

        # Define the timeframes supported by Binance
        timeframes = {
            'daily': '1d',
            'hourly': '1h'
        }
        if granularity not in timeframes:
            raise ValueError(f"Granularity '{granularity}' not supported. Use 'daily' or 'hourly'.")

        # Fetch OHLCV (Open, High, Low, Close, Volume) data
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframes[granularity], since=since)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['price'] = df['close']
        df = df[['time', 'price', 'volume']]

        if df.empty:
            print(f"Warning: No data returned for cryptocurrency {ticker}")
            return None

        return df

    except Exception as e:
        print(f"Error fetching cryptocurrency data for {ticker}: {e}")
        return None