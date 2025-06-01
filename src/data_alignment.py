import pandas as pd


def normalize_dates(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Normalize the datetime column in a DataFrame by removing time component and timezone.

    Args:
        df (pd.DataFrame): DataFrame with a date column.
        date_column (str): Column name of the date field.

    Returns:
        pd.DataFrame: DataFrame with normalized date column as datetime.date
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce').dt.tz_localize(None).dt.date
    return df


def align_datasets_by_date(news_df: pd.DataFrame, stock_df: pd.DataFrame,
                            date_column: str = 'date') -> pd.DataFrame:
    """
    Normalize and align two datasets by their date column.

    Args:
        news_df (pd.DataFrame): News headlines DataFrame
        stock_df (pd.DataFrame): Stock price DataFrame
        date_column (str): The name of the date column in both DataFrames

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of aligned news and stock DataFrames
    """
    # Ensure both datasets have a 'date' column
    news_df.rename(columns={'Date': 'date'}, inplace=True)
    stock_df.rename(columns={'Date': 'date'}, inplace=True)
    
    news_df = normalize_dates(news_df, date_column)
    stock_df = normalize_dates(stock_df, date_column)
    
    # Inner join to filter only overlapping dates
    overlapping_dates = set(news_df[date_column]) & set(stock_df[date_column])
    news_df = news_df[news_df[date_column].isin(overlapping_dates)]
    stock_df = stock_df[stock_df[date_column].isin(overlapping_dates)]

    return news_df.reset_index(drop=True), stock_df.reset_index(drop=True)
