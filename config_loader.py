def load_config(config_path="config.json"):
    """Load configuration settings from the config file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def get_unified_assets(config):
    """Combine stocks and cryptos into a single list with type labels."""
    stocks = [{"name": stock, "type": "stock"} for stock in config["stocks"]]
    cryptos = [{"name": crypto, "type": "crypto"} for crypto in config["cryptos"]]
    return stocks + cryptos