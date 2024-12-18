import yfinance as yf
import requests
import pandas as pd

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def fetch_crypto_data(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching data from Binance API: {response.status_code}")

    data = response.json()
    if not data:
        print(f"No data returned for symbol {symbol} with interval {interval}")
        return pd.DataFrame()  # Return an empty DataFrame if no data

    # Parse response into a DataFrame
    columns = ["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time"]
    df = pd.DataFrame(data, columns=columns)
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    return df