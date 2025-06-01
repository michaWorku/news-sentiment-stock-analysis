# src/correlation_analysis.py

import pandas as pd
from scipy.stats import pearsonr


def calculate_daily_returns(df: pd.DataFrame, price_column: str = 'Close', date_column: str = 'date') -> pd.DataFrame:
    """
    Calculate daily stock return as percentage change of closing price.

    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        price_column (str): Column name for closing prices
        date_column (str): Date column to sort by

    Returns:
        pd.DataFrame: DataFrame with new 'daily_return' column
    """
    df = df.sort_values(date_column).copy()
    df['daily_return'] = df[price_column].pct_change()
    return df


def merge_sentiment_with_returns(stock_df: pd.DataFrame,
                                  sentiment_df: pd.DataFrame,
                                  date_column: str = 'date') -> pd.DataFrame:
    """
    Merge stock return data and average sentiment data by date.

    Args:
        stock_df (pd.DataFrame): DataFrame with daily returns
        sentiment_df (pd.DataFrame): DataFrame with average daily sentiment
        date_column (str): Common date column to merge on

    Returns:
        pd.DataFrame: Merged DataFrame with 'daily_return' and 'avg_sentiment'
    """
    return pd.merge(stock_df, sentiment_df, on=date_column, how='inner')


def compute_correlation(df: pd.DataFrame,
                         return_col: str = 'daily_return',
                         sentiment_col: str = 'avg_sentiment') -> float:
    """
    Compute Pearson correlation between sentiment and stock returns.

    Args:
        df (pd.DataFrame): DataFrame containing sentiment and returns
        return_col (str): Column name for returns
        sentiment_col (str): Column name for sentiment

    Returns:
        float: Pearson correlation coefficient
    """
    df = df.dropna(subset=[return_col, sentiment_col])
    corr, p_value = pearsonr(df[return_col], df[sentiment_col])
    print(f"📈 Pearson correlation: {corr:.4f} (p-value: {p_value:.4f})")
    return corr
