from modules.market_data import collect_crypto_data, collect_stock_data, collect_news_data, fetch_stock_data
from modules.eda import clean_data, create_technical_indicators, add_price_change_label, merge_datasets
from modules.train import split_data, train_classifier, evaluate_model, train_random_forest, train_xgboost, ensemble_predict
from sklearn.metrics import classification_report, roc_auc_score
import os
import numpy as np
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

    print("\nAfter merging datasets:")
    print("Columns in merged dataset:", merged_data.columns)
    print("Unique crypto tickers:", merged_data["ticker_crypto"].unique())
    print("Unique stock tickers:", merged_data["ticker_stock"].unique())

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test, encoders, tickers_test = split_data(merged_data, target_col="price_change")
    
    print("\nColumns in X_test after split_data:", X_test.columns)
    if "ticker_crypto_encoded" not in X_test.columns and "ticker_stock_encoded" not in X_test.columns:
        print("Error: Encoded ticker columns are missing after split_data.")

    # Check for non-numeric columns in X_train and X_val
    if not all(np.issubdtype(dtype, np.number) for dtype in X_train.dtypes):
        print("Non-numeric columns in X_train:")
        print(X_train.select_dtypes(exclude=[np.number]).head())
        raise ValueError("X_train contains non-numeric data. Check feature preparation.")

    if not all(np.issubdtype(dtype, np.number) for dtype in X_val.dtypes):
        print("Non-numeric columns in X_val:")
        print(X_val.select_dtypes(exclude=[np.number]).head())
        raise ValueError("X_val contains non-numeric data. Check feature preparation.")

    # Debugging: Print shapes
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    # Train the Random Forest model
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)

    # Inspect Random Forest feature importance (if needed)
    rf_importance_file = "data/processed/random_forest_feature_importances.csv"
    print("Random Forest feature importances saved to", rf_importance_file)

    # Train XGBoost
    print("Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # Inspect XGBoost feature importance (already integrated in train_xgboost)
    xgb_importance_file = "data/processed/xgboost_feature_importances.csv"
    print("XGBoost feature importances saved to", xgb_importance_file)

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

    # Generate and display final prediction output

    # Predict on the test set
    ensemble_pred = ensemble_predict(rf_model, xgb_model, X_test)

    # Debugging: Check available columns
    print("\nAvailable columns in X_test before mapping tickers:", X_test.columns)

    if "ticker_crypto" not in X_test.columns:
        X_test["ticker_crypto"] = None
    if "ticker_stock" not in X_test.columns:
        X_test["ticker_stock"] = None

    # Decode crypto and stock tickers in the test set
    if "ticker_crypto_encoded" in X_test.columns:
        X_test["ticker_crypto"] = encoders["crypto"].inverse_transform(X_test["ticker_crypto_encoded"])
    if "ticker_stock_encoded" in X_test.columns:
        X_test["ticker_stock"] = encoders["stock"].inverse_transform(X_test["ticker_stock_encoded"])

    # Combine crypto and stock tickers into a unified "ticker" column
    if "ticker_crypto" in X_test.columns or "ticker_stock" in X_test.columns:
        X_test["ticker"] = X_test["ticker_crypto"].combine_first(X_test["ticker_stock"])

    # Ensure predictions align with X_test rows
    print("Shape of ensemble predictions:", ensemble_pred.shape)
    print("Shape of X_test:", X_test.shape)

    print("Raw predictions from ensemble model:")
    print(ensemble_pred)
    print("\nDistribution of predictions:")
    print(pd.Series(ensemble_pred).value_counts())

    # Add predictions and map labels
    X_test["prediction"] = ensemble_pred
    X_test["prediction_label"] = X_test["prediction"].map({0: "Down", 1: "Up"})

    # Check if prediction labels are assigned correctly
    print("\nSample of X_test with predictions:")
    # print(X_test[["ticker_crypto", "ticker_stock", "prediction", "prediction_label"]].head())
    print(X_test.head())

    # Separate predictions for cryptocurrencies and stocks
    if "ticker_crypto" in X_test.columns:
        crypto_detailed = X_test[X_test["ticker_crypto"].notna()][
            ["ticker_crypto", "prediction", "prediction_label", "price", "volume_crypto"]
        ]

    if "ticker_stock" in X_test.columns:
        stock_detailed = X_test[X_test["ticker_stock"].notna()][
            ["ticker_stock", "prediction", "prediction_label", "adj_close", "volume_stock"]
        ]

    # Debug output
    print("\nCryptocurrency detailed predictions:")
    print(crypto_detailed.head())

    print("\nStock detailed predictions:")
    print(stock_detailed.head())

    # Save and display the outputs
    crypto_output_path = "data/processed/crypto_predictions.csv"
    stock_output_path = "data/processed/stock_predictions.csv"

    if not crypto_detailed.empty:
        crypto_detailed.to_csv(crypto_output_path, index=False)
        print(f"\nCryptocurrency predictions saved to {crypto_output_path}")
        print("\nSample of cryptocurrency predictions:")
        print(crypto_detailed.head())

    if not stock_detailed.empty:
        stock_detailed.to_csv(stock_output_path, index=False)
        print(f"\nStock predictions saved to {stock_output_path}")
        print("\nSample of stock predictions:")
        print(stock_detailed.head())

if __name__ == "__main__":
    main()