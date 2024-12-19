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

def create_technical_indicators(df, price_col="price"):
    """
    Create technical indicators for time-series data.

    Args:
        df (pd.DataFrame): Input DataFrame with price column.
        price_col (str): Column name for prices.

    Returns:
        pd.DataFrame: DataFrame with additional technical indicators.
    """
    df["7_day_ma"] = df[price_col].rolling(window=7).mean()
    df["14_day_ma"] = df[price_col].rolling(window=14).mean()
    df["30_day_ma"] = df[price_col].rolling(window=30).mean()
    df["7_day_std"] = df[price_col].rolling(window=7).std()
    df["lag_1"] = df[price_col].shift(1)
    df["lag_7"] = df[price_col].shift(7)

    return df

def aggregate_news_data(df):
    """
    Aggregate news data to count articles per day.

    Args:
        df (pd.DataFrame): Input DataFrame with publishedAt column.

    Returns:
        pd.DataFrame: DataFrame with article counts per day.
    """
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    df["date"] = df["publishedAt"].dt.date
    news_counts = df.groupby("date").size().reset_index(name="news_count")
    return news_counts

def merge_datasets(crypto_df, stock_df, news_df):
    """
    Merge cryptocurrency, stock, and news datasets.

    Args:
        crypto_df (pd.DataFrame): Processed cryptocurrency DataFrame.
        stock_df (pd.DataFrame): Processed stock DataFrame.
        news_df (pd.DataFrame): Aggregated news DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame for model input.
    """
    # Ensure time columns are aligned
    crypto_df["time"] = pd.to_datetime(crypto_df["time"])
    stock_df["time"] = pd.to_datetime(stock_df["time"])
    news_df["date"] = pd.to_datetime(news_df["date"])

    # Merge crypto and stock data on "time"
    combined = pd.merge(crypto_df, stock_df, on="time", how="inner")

    # Merge news data on "date"
    combined["date"] = combined["time"].dt.date
    combined = pd.merge(combined, news_df, on="date", how="left")

    # Fill missing news counts with 0
    combined["news_count"] = combined["news_count"].fillna(0)

    return combined

if __name__ == "__main__":
    # Load and process cryptocurrency data
    bitcoin_data = load_data("data/raw/bitcoin_data.csv")
    bitcoin_data = clean_data(bitcoin_data, required_columns=["time", "price"])
    bitcoin_data = create_technical_indicators(bitcoin_data)
    processed_bitcoin_path = os.path.join(BASE_DIR, "data/processed/bitcoin_data.csv")
    bitcoin_data.to_csv(processed_bitcoin_path, index=False)
    print(f"Processed Bitcoin data saved to {processed_bitcoin_path}")

    # Load and process stock data
    aapl_data = load_data("data/raw/AAPL_stock_data.csv")
    aapl_data = clean_data(aapl_data, required_columns=["time", "adj_close"])
    aapl_data = create_technical_indicators(aapl_data, price_col="adj_close")
    processed_aapl_path = os.path.join(BASE_DIR, "data/processed/AAPL_stock_data.csv")
    aapl_data.to_csv(processed_aapl_path, index=False)
    print(f"Processed AAPL stock data saved to {processed_aapl_path}")

    # Load and process news data
    news_data = load_data("data/raw/cryptocurrency_news_data.csv")
    news_counts = aggregate_news_data(news_data)
    processed_news_path = os.path.join(BASE_DIR, "data/processed/cryptocurrency_news_data.csv")
    news_counts.to_csv(processed_news_path, index=False)
    print(f"Processed news data saved to {processed_news_path}")

    # Merge all datasets
    merged_data = merge_datasets(bitcoin_data, aapl_data, news_counts)
    merged_data_path = os.path.join(BASE_DIR, "data/processed/merged_data.csv")
    merged_data.to_csv(merged_data_path, index=False)
    print(f"Merged dataset saved to {merged_data_path}")