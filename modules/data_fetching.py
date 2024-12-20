def fetch_data(asset, lookback_days, granularity):
    """Fetch data for a given asset (stock or crypto)."""
    print(f"Fetching {lookback_days} days of data for {asset['name']}...")

    if asset["type"] == "crypto":
        # Crypto-specific API fetching logic
        df = fetch_crypto_data(asset["name"], lookback_days, granularity)
        df["ticker"] = asset["name"]
    elif asset["type"] == "stock":
        # Stock-specific API fetching logic
        df = fetch_stock_data(asset["name"], lookback_days, granularity)
        df["ticker"] = asset["name"]
    else:
        raise ValueError(f"Unsupported asset type: {asset['type']}")

    return df