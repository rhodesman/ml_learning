from modules.market_data import collect_crypto_data, collect_stock_data, collect_news_data
from modules.eda import clean_data, create_technical_indicators, add_price_change_label, merge_datasets
from modules.train import split_data, train_classifier, evaluate_model
import os
import pandas as pd

# Define paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MERGED_FILE = os.path.join(PROCESSED_DIR, "merged_data.csv")

def main():
    # Step 1: Data Collection
    print("Collecting data...")
    collect_crypto_data()
    collect_stock_data()
    collect_news_data()

    # Step 2: Data Cleaning and Feature Engineering
    print("Processing data...")
    # Load raw data
    bitcoin_data = pd.read_csv(os.path.join(RAW_DIR, "bitcoin_data.csv"))
    aapl_data = pd.read_csv(os.path.join(RAW_DIR, "AAPL_stock_data.csv"))
    news_data = pd.read_csv(os.path.join(RAW_DIR, "cryptocurrency_news_data.csv"))

    # Clean and process
    bitcoin_data = clean_data(bitcoin_data, required_columns=["time", "price"])
    bitcoin_data = create_technical_indicators(bitcoin_data)

    aapl_data = clean_data(aapl_data, required_columns=["time", "adj_close"])
    aapl_data = create_technical_indicators(aapl_data, price_col="adj_close")

    # Ensure 'publishedAt' is converted to datetime
    news_data["publishedAt"] = pd.to_datetime(news_data["publishedAt"], errors="coerce")

    # Handle rows where conversion failed (if any)
    news_data = news_data.dropna(subset=["publishedAt"])

    # Group by date
    news_counts = news_data.groupby(news_data["publishedAt"].dt.date).size().reset_index(name="news_count")
    news_counts.rename(columns={"publishedAt": "date"}, inplace=True)

    # Merge datasets
    merged_data = merge_datasets(bitcoin_data, aapl_data, news_counts)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged_data.to_csv(MERGED_FILE, index=False)
    print(f"Merged data saved to {MERGED_FILE}")

    # Step 3: Model Training
    print("Training model...")
    merged_data = add_price_change_label(merged_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_data, target_col="price_change")
    model = train_classifier(X_train, y_train, X_val, y_val)

    # Step 4: Model Evaluation
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()