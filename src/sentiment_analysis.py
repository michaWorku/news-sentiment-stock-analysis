# src/sentiment_analysis.py

import pandas as pd
from textblob import TextBlob

def get_sentiment(text: str) -> float:
    """
    Calculate sentiment polarity using TextBlob.
    
    Args:
        text (str): Input news text (e.g. headline)

    Returns:
        float: Sentiment polarity score between -1.0 (negative) to +1.0 (positive)
    """
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def apply_sentiment_analysis(df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame:
    """
    Add sentiment scores to a DataFrame using TextBlob.

    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing the headline

    Returns:
        pd.DataFrame: Original DataFrame with new 'sentiment' column
    """
    df['sentiment'] = df[text_column].apply(get_sentiment)
    return df

def aggregate_daily_sentiment(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Aggregate average sentiment by date.

    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'sentiment' columns
        date_column (str): Column name containing date information

    Returns:
        pd.DataFrame: Aggregated sentiment per day
    """
    return df.groupby(date_column)['sentiment'].mean().reset_index(name='avg_sentiment')
