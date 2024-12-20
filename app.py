from modules.data_fetcher import fetch_stock_data, fetch_crypto_data
from modules.preprocessor import preprocess_data
from modules.train import train_model
from modules.evaluation import evaluate_model, save_feature_importance


import os
import numpy as np
import pandas as pd
import json

# Load configuration
with open('configs/config.json', 'r') as f:
    config = json.load(f)

lookback_days = config['lookback_days']
assets = config['stocks'] + config['cryptos']
news_query = config['news_query']
granularity = config['granularity']



def main():
    # Fetch and combine data
    data_frames = []
    for asset in assets:
        if asset in config['stocks']:
            data = fetch_stock_data(asset, lookback_days, granularity)
        elif asset in config['cryptos']:
            data = fetch_crypto_data(asset, lookback_days, granularity)
        else:
            raise ValueError(f"Asset {asset} not recognized.")

        # Debugging: Check the fetched data
        if data is None:
            raise ValueError(f"Failed to fetch data for asset: {asset}")
        print(f"Fetched data for {asset}:")
        print(data.head())

        data['ticker'] = asset  # Ensure this is only assigned to valid data
        data_frames.append(data)

    combined_data = pd.concat(data_frames, ignore_index=True)

    # Preprocess the combined data
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(combined_data)

    # Split the preprocessed data into train/val/test sets
    X_train, X_val, y_train, y_val = split_data(preprocessed_data)
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluate_model(model, X_val, y_val)

    # Save feature importances
    save_feature_importance(model, 'data/processed/feature_importances.csv')
    

if __name__ == "__main__":
    main()