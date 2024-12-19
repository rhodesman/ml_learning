import pandas as pd
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def clean_data(df, required_columns):
    """
    Clean the data by dropping rows with missing values in required columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of columns to check for missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    return df.dropna(subset=required_columns)


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
    visualize_crypto_prices(bitcoin_data, coin_name="Bitcoin")

    # Load and inspect stock data
    aapl_data = load_data("data/raw/AAPL_stock_data.csv")
    aapl_data = clean_data(aapl_data, required_columns=["time", "adj_close"])
    visualize_stock_prices(aapl_data, ticker="AAPL")

    # Load and inspect news data
    news_data = load_data("data/raw/cryptocurrency_news_data.csv")
    analyze_news_volume(news_data, query="Cryptocurrency")