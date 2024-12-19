from modules.market_data import collect_crypto_data, collect_stock_data, collect_news_data, fetch_stock_data
from modules.eda import clean_data, create_technical_indicators, add_price_change_label, merge_datasets
from modules.train import split_data, train_classifier, evaluate_model, train_random_forest, train_xgboost, ensemble_predict
from sklearn.metrics import classification_report, roc_auc_score
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

    # Process each stock dynamically from the config
    for stock in config["stocks"]:
        print(f"Processing data for stock: {stock}")
        
        # Fetch data for the current stock
        stock_data = fetch_stock_data(ticker=stock, days=config["lookback_days"])
        
        # Clean the stock data
        print(f"Cleaning data for {stock}...")
        stock_data = clean_data(stock_data, required_columns=["time", "adj_close"])
        
        # Add technical indicators
        print(f"Adding technical indicators to {stock} data...")
        stock_data = create_technical_indicators(stock_data, price_col="adj_close")
        
        # Save or process stock_data as needed
        print(f"Finished processing {stock}.\n")

    # Ensure 'publishedAt' is converted to datetime
    news_data["publishedAt"] = pd.to_datetime(news_data["publishedAt"], errors="coerce")

    # Handle rows where conversion failed (if any)
    news_data = news_data.dropna(subset=["publishedAt"])

    # Group by date
    news_counts = news_data.groupby(news_data["publishedAt"].dt.date).size().reset_index(name="news_count")
    news_counts.rename(columns={"publishedAt": "date"}, inplace=True)

    # Merge datasets
    merged_data = merge_datasets(bitcoin_data, stock_data, news_counts)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged_data.to_csv(MERGED_FILE, index=False)
    #print(f"Merged data saved to {MERGED_FILE}")

   # Step 3: Model Training
    print("Training models...")

    # Add labels for price change
    merged_data = add_price_change_label(merged_data)

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test, encoders = split_data(merged_data, target_col="price_change")

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

    print("\nTickers in test set:")
    print(X_test["ticker"].value_counts())
    # Generate and display final prediction output
    # Predict on the test set
    ensemble_pred = ensemble_predict(rf_model, xgb_model, X_test)

    # Map predictions back to stock/crypto names
    if "ticker_crypto_encoded" in X_test.columns:
        X_test["crypto_ticker"] = encoders["crypto"].inverse_transform(X_test["ticker_crypto_encoded"])
    if "ticker_stock_encoded" in X_test.columns:
        X_test["stock_ticker"] = encoders["stock"].inverse_transform(X_test["ticker_stock_encoded"])

    # Add predictions to the dataset for interpretation
    X_test["prediction"] = ensemble_pred
    X_test["prediction_label"] = X_test["prediction"].map({0: "Down", 1: "Up"})

    # Ensure ticker column is present
    if "ticker_crypto" in X_test.columns or "ticker_stock" in X_test.columns:
        X_test["ticker"] = X_test["ticker_crypto"].combine_first(X_test["ticker_stock"])
        print("\nTickers in test set:")
        print(X_test["ticker"].value_counts())
    else:
        print("Warning: No ticker information available in test set.")

    # Display prediction results grouped by ticker
    print("\nPrediction Results:")
    prediction_summary = X_test.groupby(["crypto_ticker", "stock_ticker", "prediction_label"]).size().reset_index(name="Count")
    print(prediction_summary)

    # Optionally, save the predictions to a file
    prediction_output_path = "data/processed/prediction_results.csv"
    X_test.to_csv(prediction_output_path, index=False)
    print(f"\nPredictions saved to {prediction_output_path}")

if __name__ == "__main__":
    main()