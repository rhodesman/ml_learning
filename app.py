from modules.market_data import fetch_stock_data, fetch_crypto_data
from modules.news_data import fetch_news
from modules.preprocessing import clean_market_data, clean_news_data
import os

# Create necessary directories if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Step 1: Collect Stock Data
print("Fetching stock data...")
stock_data = fetch_stock_data("AAPL", start="2020-01-01", end="2023-12-31")
stock_data.to_csv("data/raw/aapl_stock_data.csv")
print("Stock data saved to data/raw/aapl_stock_data.csv")

# Step 2: Collect Crypto Data
# Fetch crypto data using Coinbase API
print("Fetching crypto data...")
crypto_data = fetch_crypto_data(coin_id="ethereum", vs_currency="usd", days=7)
crypto_data.to_csv("data/raw/debug_crypto_data.csv", index=False)
print("Crypto data fetched and saved to data/raw/debug_crypto_data.csv.")

# Step 3: Collect News Data
print("Fetching news data...")
news_data = fetch_news("politics")
print("News data fetched. Titles:")
for news in news_data:
    print(news["title"])

#debug returned data
print(crypto_data.head())
print(crypto_data.dtypes)
crypto_data.to_csv("data/raw/debug_crypto_data.csv")
print("Debug crypto data saved for inspection.")

# Step 4: Preprocess Stock Data
print("Preprocessing stock data...")
cleaned_stocks = clean_market_data(stock_data)
cleaned_stocks.to_csv("data/processed/cleaned_stock_data.csv")
print("Cleaned stock data saved to data/processed/cleaned_stock_data.csv")

# Step 5: Preprocess Crypto Data
print("Preprocessing crypto data...")
cleaned_crypto = clean_market_data(crypto_data)
cleaned_crypto.to_csv("data/processed/cleaned_crypto_data.csv")
print("Cleaned crypto data saved to data/processed/cleaned_crypto_data.csv")

# Step 6: Preprocess News Data
print("Preprocessing news data...")
cleaned_news = clean_news_data(news_data)
print("Cleaned news titles:")
for news in cleaned_news:
    print(news["title"])