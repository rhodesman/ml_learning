import pandas as pd

# Load the processed stock data
stock_data = pd.read_csv("data/processed/cleaned_stock_data.csv")

# Display the first few rows and data types
print(stock_data.head())
print(stock_data.dtypes)