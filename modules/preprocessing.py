import pandas as pd

def clean_market_data(df):
    # Remove unnecessary columns, fill missing values, etc.
    df = df.dropna()
    return df

def clean_news_data(news_list):
    # Perform text preprocessing like tokenization, lowercasing
    return [{"title": news["title"].lower(), "link": news["link"]} for news in news_list]