import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_market_data(df):
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The input DataFrame is empty. Check your data source.")

    # Ensure only numerical columns are selected for scaling
    required_columns = ["Open", "High", "Low", "Close"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Drop any non-numeric rows or NaN values
    df = df.dropna(subset=required_columns)

    # Normalize columns using MinMaxScaler
    scaler = MinMaxScaler()
    for column in required_columns:
        df[column] = scaler.fit_transform(df[[column]])

    return df

def clean_news_data(news_list):
    # Perform text preprocessing like tokenization, lowercasing
    return [{"title": news["title"].lower(), "link": news["link"]} for news in news_list]