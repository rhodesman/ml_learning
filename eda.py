import pandas as pd
import matplotlib.pyplot as plt

# Load processed stock data
stock_data = pd.read_csv("data/processed/cleaned_stock_data.csv")

# Plot closing prices
plt.figure(figsize=(10, 6))
plt.plot(stock_data["Close"])
plt.title("Stock Closing Prices")
plt.xlabel("Time")
plt.ylabel("Normalized Price")
plt.show()