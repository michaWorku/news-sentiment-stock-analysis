import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesAnalysis:
    """
    A module for analyzing time-based trends in financial news articles.
    """

    def __init__(self, filepath: str):
        """
        Initialize the class and load the dataset.

        Args:
            filepath (str): Path to the news dataset CSV.
        """
        self.filepath = filepath
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load and validate CSV file.

        Returns:
            pd.DataFrame: Loaded dataset
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        df = pd.read_csv(self.filepath)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df

    def plot_article_frequency_over_time(self):
        """
        Plot the number of articles published per day.
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

    def plot_article_distribution_by_hour(self):
        """
        Plot the distribution of article publication times by hour.
        """
        self.df['hour'] = self.df['date'].dt.hour
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['hour'], bins=24, kde=True)
        plt.title("🕒 Distribution of Article Publication by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def top_publishers(self, top_n: int = 10):
        """
        Print and return the top publishers by article count.

        Args:
            top_n (int): Number of top publishers to return.

        Returns:
            pd.Series: Publisher counts
        """
        publishers = self.df['publisher'].value_counts().head(top_n)
        print("\n🏢 Top Publishers:")
        print(publishers)
        return publishers

    def top_email_domains(self, top_n: int = 10):
        """
        Identify top email domains from publisher field.

        Args:
            top_n (int): Number of top domains to return.

        Returns:
            pd.Series: Top email domains
        """
        email_publishers = self.df['publisher'].dropna().astype(str)
        domains = email_publishers[email_publishers.str.contains('@')].apply(lambda x: x.split('@')[-1])
        domain_counts = domains.value_counts().head(top_n)
        print("\n📧 Top Email Domains:")
        print(domain_counts)
        return domain_counts


# Example usage:
# tsa = TimeSeriesAnalysis("../extracted_data/raw_analyst_ratings.csv")
# tsa.plot_article_frequency_over_time()
# tsa.plot_article_distribution_by_hour()
# tsa.top_publishers()
# tsa.top_email_domains()
