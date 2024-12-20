import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the data for machine learning models.

    Args:
        df (pd.DataFrame): The combined raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset ready for model training.
    """
    # Handle missing values
    print("Handling missing values...")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Feature engineering: Adding moving averages
    print("Adding moving averages...")
    for window in [7, 14, 30]:
        df[f'{window}_day_ma'] = df['price'].rolling(window=window).mean()

    # Lag features
    print("Adding lag features...")
    df['lag_1'] = df['price'].shift(1)
    df['lag_7'] = df['price'].shift(7)

    # Drop rows with NaN values after feature engineering
    print("Dropping rows with NaN values after feature engineering...")
    df.dropna(inplace=True)

    # Normalize numerical features
    print("Normalizing features...")
    scaler = StandardScaler()
    numeric_columns = ['price', 'volume'] + [col for col in df.columns if 'ma' in col or 'lag' in col]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Encode tickers as numeric values
    print("Encoding tickers...")
    df['ticker_encoded'] = pd.factorize(df['ticker'])[0]

    return df