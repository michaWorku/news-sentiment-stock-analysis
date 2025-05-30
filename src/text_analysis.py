import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

class TextAnalyzer:
    """
    A module for keyword frequency and topic exploration from text data.
    """

    def __init__(self, filepath: str):
        """
        Initialize the class and load dataset.

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
        return pd.read_csv(self.filepath)

    def generate_wordcloud(self):
        """
        Generate and display a word cloud from headlines.
        """
        text = ' '.join(self.df['headline'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("☁️ Word Cloud of Headlines")
        plt.show()

    def get_top_keywords(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Extract the most frequent keywords using CountVectorizer.

        Args:
            top_n (int): Number of top keywords to return.

        Returns:
            List[Tuple[str, int]]: List of (keyword, frequency)
        """
        headlines = self.df['headline'].dropna().astype(str)
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(headlines)
        word_counts = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        freq_dict = dict(zip(vocab, word_counts))
        return sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def plot_top_keywords(self, top_n: int = 20):
        """
        Plot a bar chart of top keywords.

        Args:
            top_n (int): Number of top keywords to plot.
        """
        keywords = self.get_top_keywords(top_n)
        words, freqs = zip(*keywords)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(freqs), y=list(words), palette='viridis')
        plt.title(f"🔑 Top {top_n} Keywords in Headlines")
        plt.xlabel("Frequency")
        plt.ylabel("Keyword")
        plt.tight_layout()
        plt.show()


# Example Usage:
# analyzer = TextAnalyzer("../extrated_data/raw_analyst_ratings.csv")
# analyzer.generate_wordcloud()
# print(analyzer.get_top_keywords())
# analyzer.plot_top_keywords()
