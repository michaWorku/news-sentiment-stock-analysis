import pandas as pd
import os
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class DescriptiveStats:
    """
    Module for descriptive statistics on financial news data.
    """

    def __init__(self, filepath: str):
        """
        Initialize and load the dataset.
        
        Args:
            filepath (str): Path to the raw dataset.
        """
        self.filepath = filepath
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load the CSV file and parse the date.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        df = pd.read_csv(self.filepath)
        
        # Explicitly convert date column to datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df


    def headline_length_statistics(self) -> pd.Series:
        """
        Compute basic statistics for the headline text length.
        
        Returns:
            pd.Series: Descriptive statistics.
        """
        self.df['headline_length'] = self.df['headline'].astype(str).apply(len)
        return self.df['headline_length'].describe()

    def articles_per_publisher(self) -> pd.Series:
        """
        Count articles by publisher.
        
        Returns:
            pd.Series: Publisher frequency.
        """
        return self.df['publisher'].value_counts()

    def plot_publication_trends(self):
        """
        Plot number of articles published per day.
        """
        self.df['date_only'] = self.df['date'].dt.date
        daily_counts = self.df['date_only'].value_counts().sort_index()

        plt.figure(figsize=(12, 5))
        sns.lineplot(x=daily_counts.index, y=daily_counts.values)
        plt.title("📈 Article Publication Frequency Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example Usage
# stats = DescriptiveStats("../extracted_data/raw_analyst_ratings.csv")
# print(stats.headline_length_statistics())
# print(stats.articles_per_publisher().head(10))
# stats.plot_publication_trends()
