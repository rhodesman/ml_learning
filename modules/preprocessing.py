import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_market_data(df):
    # Drop any non-numeric columns or extra rows
    df = df.iloc[1:]  # Skip the extra header row
    df = df.apply(pd.to_numeric, errors="coerce")  # Convert columns to numeric

    # Normalize relevant columns
    scaler = MinMaxScaler()
    for column in ["Open", "High", "Low", "Close"]:
        if column in df.columns:
            df[column] = scaler.fit_transform(df[[column]])

    # Drop rows with NaN values
    df = df.dropna()

    return df

def clean_news_data(news_list):
    # Perform text preprocessing like tokenization, lowercasing
    return [{"title": news["title"].lower(), "link": news["link"]} for news in news_list]