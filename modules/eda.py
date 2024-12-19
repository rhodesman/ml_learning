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
    Clean the dataset by dropping rows with missing required columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of required column names.

    Returns:
        pd.DataFrame: Cleaned DataFrame with no missing values in required columns.
    """
    # Check for columns that start with required column names
    column_map = {col: [c for c in df.columns if c.startswith(col)] for col in required_columns}

    resolved_columns = []
    for key, matches in column_map.items():
        if matches:
            resolved_columns.append(matches[0])  # Use the first matching column
        else:
            print(f"Available columns: {df.columns}")  # Debugging print
            raise KeyError(f"Required column '{key}' not found in the dataset.")

    print(f"Resolved Columns: {resolved_columns}")  # Debugging print

    # Drop rows with missing values in the resolved columns
    df = df.dropna(subset=resolved_columns)

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
    Add technical indicators to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        price_col (str): The base name of the column representing the price.

    Returns:
        pd.DataFrame: The DataFrame with additional technical indicators.
    """
    # Resolve the actual price column name
    matching_columns = [col for col in df.columns if col.startswith(price_col)]
    if not matching_columns:
        raise KeyError(f"Column '{price_col}' not found in the DataFrame.")
    resolved_price_col = matching_columns[0]

    print(f"Resolved price column: {resolved_price_col}")  # Debugging print

    # Add rolling averages and other technical indicators
    df["7_day_ma"] = df[resolved_price_col].rolling(window=7).mean()
    df["14_day_ma"] = df[resolved_price_col].rolling(window=14).mean()
    df["30_day_ma"] = df[resolved_price_col].rolling(window=30).mean()
    df["7_day_std"] = df[resolved_price_col].rolling(window=7).std()

    # Add lag features
    df["lag_1"] = df[resolved_price_col].shift(1)
    df["lag_7"] = df[resolved_price_col].shift(7)

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
    Merge cryptocurrency, stock, and news datasets on a common time column.

    Args:
        crypto_df (pd.DataFrame): Cryptocurrency data.
        stock_df (pd.DataFrame): Stock market data.
        news_df (pd.DataFrame): News data.

    Returns:
        pd.DataFrame: Merged dataset with ticker information.
    """
    # Resolve the time column in stock_df
    time_column = next((col for col in stock_df.columns if "time" in col.lower()), None)
    if not time_column:
        raise KeyError("No time column found in the stock dataset.")
    print(f"Resolved time column in stock data: {time_column}")  # Debugging print

    # Resolve the time column in news_df
    news_time_column = next((col for col in news_df.columns if "time" in col.lower() or "date" in col.lower()), None)
    if not news_time_column:
        raise KeyError("No time or date column found in the news dataset.")
    print(f"Resolved time column in news data: {news_time_column}")  # Debugging print

    # Rename columns for consistency
    stock_df.rename(columns={time_column: "time"}, inplace=True)
    news_df.rename(columns={news_time_column: "time"}, inplace=True)

    # Convert time columns to datetime
    crypto_df["time"] = pd.to_datetime(crypto_df["time"])
    stock_df["time"] = pd.to_datetime(stock_df["time"])
    news_df["time"] = pd.to_datetime(news_df["time"])

    # Add 'ticker' column to stock_df if missing
    if "ticker" not in stock_df.columns:
        stock_df["ticker"] = "stock"  # Placeholder if no ticker provided

    # Add 'ticker' column to crypto_df
    if "ticker" not in crypto_df.columns:
        crypto_df["ticker"] = "crypto"  # Placeholder if no ticker provided

    # Merge datasets
    merged = pd.merge(crypto_df, stock_df, on="time", how="inner", suffixes=("_crypto", "_stock"))
    merged = pd.merge(merged, news_df, on="time", how="left")

    # Debugging merged dataset
    print("\nAfter merging datasets:")
    print("Columns in merged dataset:", merged.columns)
    if "ticker_crypto" in merged.columns:
        print("Unique crypto tickers:", merged["ticker_crypto"].unique())
    if "ticker_stock" in merged.columns:
        print("Unique stock tickers:", merged["ticker_stock"].unique())

    return merged

def add_price_change_label(df, price_col="price"):
    """
    Add a binary column indicating if the price increased the next day.

    Args:
        df (pd.DataFrame): Input DataFrame with price data.
        price_col (str): Column name for the price.

    Returns:
        pd.DataFrame: DataFrame with added 'price_change' column.
    """
    df["next_day_price"] = df[price_col].shift(-1)
    df["price_change"] = (df["next_day_price"] > df[price_col]).astype(int)
    df = df.drop(columns=["next_day_price"])  # Remove helper column
    return df


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