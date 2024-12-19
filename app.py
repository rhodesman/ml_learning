from modules.market_data import collect_crypto_data, collect_stock_data, collect_news_data
from modules.eda import clean_data, create_technical_indicators, add_price_change_label, merge_datasets
from modules.train import split_data, train_classifier, evaluate_model
import os
import pandas as pd
import json

# Load configuration
with open("configs/config.json", "r") as f:
    config = json.load(f)

LOOKBACK_DAYS = config["lookback_days"]
STOCKS = config["stocks"]
CRYPTOS = config["cryptos"]
NEWS_QUERY = config["news_query"]
GRANULARITY = config["granularity"]

# Define paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MERGED_FILE = os.path.join(PROCESSED_DIR, "merged_data.csv")

def main():
    # Step 1: Data Collection
    #print("Collecting data...")
    collect_crypto_data(config["cryptos"], config["lookback_days"])
    collect_stock_data(config["stocks"], config["lookback_days"])
    collect_news_data(config["news_query"], config["lookback_days"])

    # Step 2: Data Cleaning and Feature Engineering
    #print("Processing data...")
    # Load raw data
    bitcoin_data = pd.read_csv(os.path.join(RAW_DIR, "bitcoin_data.csv"))
    aapl_data = pd.read_csv(os.path.join(RAW_DIR, "AAPL_stock_data.csv"))
    news_data = pd.read_csv(os.path.join(RAW_DIR, "cryptocurrency_news_data.csv"))

    # Clean and process
    bitcoin_data = clean_data(bitcoin_data, required_columns=["time", "price"])
    bitcoin_data = create_technical_indicators(bitcoin_data)

    #aapl_data = clean_data(aapl_data, required_columns=["time", "adj_close"])
    # Clean stock data
    print("Cleaning AAPL data...")
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
    #print(f"Merged data saved to {MERGED_FILE}")

   # Step 3: Model Training
    print("Training models...")

    # Add labels for price change
    merged_data = add_price_change_label(merged_data)

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_data, target_col="price_change")

    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)

    # Train XGBoost
    print("Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # Step 4: Ensemble Evaluation
    print("Evaluating Ensemble...")

    # Get probabilities for the test set
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Random Forest probabilities
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # XGBoost probabilities

    # Average the probabilities
    ensemble_pred_proba = (rf_pred_proba + xgb_pred_proba) / 2
    ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)

    # Evaluate Ensemble Performance
    print("Ensemble Model Performance:")
    print(classification_report(y_test, ensemble_pred, zero_division=0))

    # Compute AUC-ROC
    auc = roc_auc_score(y_test, ensemble_pred_proba)
    print(f"Ensemble AUC-ROC: {auc}")

if __name__ == "__main__":
    main()