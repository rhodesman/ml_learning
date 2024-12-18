import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_market_data(df):
    # Ensure valid numerical columns
    required_columns = ["Open", "High", "Low", "Close"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
    
    # Drop non-numeric rows and extra headers
    df = df.iloc[1:]  # Skip the extra header row if present
    df[required_columns] = df[required_columns].apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaN values in required columns
    df = df.dropna(subset=required_columns)

    # Normalize the data
    scaler = MinMaxScaler()
    for column in required_columns:
        df[column] = scaler.fit_transform(df[[column]])

    return df

def clean_news_data(news_list):
    # Perform text preprocessing like tokenization, lowercasing
    return [{"title": news["title"].lower(), "link": news["link"]} for news in news_list]