import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # One level up from 'modules'

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    full_path = os.path.join(BASE_DIR, file_path)  # Construct full path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    df = pd.read_csv(full_path)
    print(f"Loaded data from {file_path}:")
    print(df.head())
    return df


def clean_data(df, required_columns):
    """
    Clean the data by dropping rows with missing values in required columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of columns to check for missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    print("Before cleaning:")
    print(df.info())

    # Drop rows where required columns are missing
    df = df.dropna(subset=required_columns)

    print("After cleaning:")
    print(df.info())
    return df


def visualize_crypto_prices(df, coin_name="Bitcoin"):
    """
    Plot cryptocurrency prices over time.

    Args:
        df (pd.DataFrame): DataFrame containing time and price columns.
        coin_name (str): Name of the cryptocurrency for the title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(df["time"]), df["price"], label=f"{coin_name} Price")
    plt.title(f"{coin_name} Price Over Time")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_stock_prices(df, ticker="AAPL"):
    """
    Plot stock prices over time.

    Args:
        df (pd.DataFrame): DataFrame containing time and adj_close columns.
        ticker (str): Stock ticker for the title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(df["time"]), df["adj_close"], label=f"{ticker} Adjusted Close Price")
    plt.title(f"{ticker} Stock Price Over Time")
    plt.xlabel("Time")
    plt.ylabel("Adjusted Close Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()


def analyze_news_volume(df, query="Cryptocurrency"):
    """
    Analyze and plot the volume of news articles over time.

    Args:
        df (pd.DataFrame): DataFrame containing publishedAt column.
        query (str): Query term for the title.
    """
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    df["date"] = df["publishedAt"].dt.date
    news_volume = df.groupby("date").size()

    plt.figure(figsize=(10, 6))
    news_volume.plot(kind="bar", color="orange", alpha=0.7)
    plt.title(f"Volume of News Articles for '{query}' Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.grid(axis="y")
    plt.show()


if __name__ == "__main__":
    # Load and inspect cryptocurrency data
    bitcoin_data = load_data("data/raw/bitcoin_data.csv")
    bitcoin_data = clean_data(bitcoin_data, required_columns=["time", "price"])

    # Save cleaned Bitcoin data
    processed_bitcoin_path = os.path.join(BASE_DIR, "data/processed/bitcoin_data.csv")
    bitcoin_data.to_csv(processed_bitcoin_path, index=False)
    print(f"Cleaned Bitcoin data saved to {processed_bitcoin_path}")

    # Visualize Bitcoin data
    visualize_crypto_prices(bitcoin_data, coin_name="Bitcoin")

    # Load and inspect stock data
    aapl_data = load_data("data/raw/AAPL_stock_data.csv")
    aapl_data = clean_data(aapl_data, required_columns=["time", "adj_close"])

    # Save cleaned Apple stock data
    processed_aapl_path = os.path.join(BASE_DIR, "data/processed/AAPL_stock_data.csv")
    aapl_data.to_csv(processed_aapl_path, index=False)
    print(f"Cleaned AAPL stock data saved to {processed_aapl_path}")

    # Visualize Apple stock data
    visualize_stock_prices(aapl_data, ticker="AAPL")

    # Load and inspect news data
    news_data = load_data("data/raw/cryptocurrency_news_data.csv")
    news_data = clean_data(news_data, required_columns=["publishedAt"])

    # Save cleaned news data
    processed_news_path = os.path.join(BASE_DIR, "data/processed/cryptocurrency_news_data.csv")
    news_data.to_csv(processed_news_path, index=False)
    print(f"Cleaned news data saved to {processed_news_path}")

    # Visualize news data
    analyze_news_volume(news_data, query="Cryptocurrency")