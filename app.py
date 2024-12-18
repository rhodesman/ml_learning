import json
from modules.market_data import fetch_stock_data, fetch_crypto_data
from modules.news_data import fetch_news
from modules.preprocessing import clean_market_data, clean_news_data

# Load configurations
with open("configs/config.json") as f:
    config = json.load(f)

# Fetch Data
stock_data = fetch_stock_data("AAPL", "2020-01-01", "2023-12-31")
crypto_data = fetch_crypto_data("BTCUSDT", "1d", 100)
news_data = fetch_news(config["news_query"])

# Preprocess Data
cleaned_stocks = clean_market_data(stock_data)
cleaned_news = clean_news_data(news_data)

# Output Results
print(cleaned_stocks.head())
print(cleaned_news)