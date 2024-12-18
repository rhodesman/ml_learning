import yfinance as yf
import requests
import pandas as pd

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def fetch_crypto_data(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url).json()
    columns = ["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time"]
    df = pd.DataFrame(response, columns=columns)
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    return df