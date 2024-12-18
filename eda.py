import pandas as pd
import matplotlib.pyplot as plt

# Load the processed stock data, skipping the extra header row
stock_data = pd.read_csv("data/processed/cleaned_stock_data.csv", skiprows=1)

# Ensure numerical columns are properly parsed
stock_data["Close"] = pd.to_numeric(stock_data["Close"], errors="coerce")

# Drop rows with NaN values
stock_data = stock_data.dropna(subset=["Close"])

# Plot closing prices
plt.figure(figsize=(10, 6))
plt.plot(stock_data["Close"], label="Closing Price")
plt.title("Stock Closing Prices")
plt.xlabel("Time")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()