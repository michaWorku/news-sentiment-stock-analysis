import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TechnicalAnalysis:
    """
    TechnicalAnalysis is a comprehensive utility class for performing stock market
    data analysis using historical and live data. It includes:

    - Loading and preprocessing stock datasets (CSV or yfinance).
    - Calculation of technical indicators (Moving Averages, RSI, MACD).
    - Visualization of indicators and volume data.
    - Summary statistics for financial insight.

    Usage:
        ta = TechnicalAnalysis()
        ta.fetch_from_yfinance('AAPL')
        ta.clean_data()
        ta.calculate_indicators()
        ta.plot_stock_data()
    """

    def __init__(self, filepath):
        """
        Initialize the analysis by loading one or multiple stock CSV files.

        Args:
            filepath (str or list): File path, list of file paths, or directory path containing CSVs.
        """
        self.filepath = filepath
        if filepath != None:
            self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load and concatenate stock data from one or multiple CSV files.

        Returns:
            pd.DataFrame: Combined stock price data.
        """
        file_list = []

        if isinstance(self.filepath, str):
            file_list = glob.glob(os.path.join(self.filepath, '*.csv')) if os.path.isdir(self.filepath) else [self.filepath]
        elif isinstance(self.filepath, list):
            file_list = self.filepath

        dataframes = []
        for file in file_list:
            df = pd.read_csv(file)
            name = os.path.splitext(os.path.basename(file))[0].upper()
            df['Company'] = name
            dataframes.append(df)

        full_df = pd.concat(dataframes, ignore_index=True)
        os.makedirs("../data/extracted", exist_ok=True)
        full_df.to_csv("../data/extracted/combined_stocks.csv", index=False)
        return full_df

    def clean_data(self, date_column='date'):
        """
        Clean and standardize date column and remove null entries.

        Args:
            date_column (str): Name of the column to treat as date.
        """
        
        if date_column in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce').dt.tz_localize(None)
        self.df.dropna(inplace=True)

    def calculate_indicators(self):
        """
        Calculate technical indicators using TA-Lib: MA20, MA50, RSI, and MACD.
        """
        self.df['MA20'] = talib.SMA(self.df['Close'], timeperiod=20)
        self.df['MA50'] = talib.SMA(self.df['Close'], timeperiod=50)
        self.df['RSI'] = talib.RSI(self.df['Close'], timeperiod=14)
        self.df['MACD'], self.df['MACD_signal'], _ = talib.MACD(
            self.df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

    def plot_moving_averages(self):
        """
        Plot 20-day and 50-day moving averages along with the closing price.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Close'], label='Close Price')
        plt.plot(self.df['MA20'], label='MA 20')
        plt.plot(self.df['MA50'], label='MA 50')
        plt.title("📈 Moving Averages")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_rsi(self):
        """
        Plot the Relative Strength Index (RSI) with overbought/oversold thresholds.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.df['RSI'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.axhline(30, linestyle='--', color='green', alpha=0.5)
        plt.title("📊 Relative Strength Index (RSI)")
        plt.xlabel("Time")
        plt.ylabel("RSI")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_macd(self):
        """
        Plot MACD and signal line to show momentum trends.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(self.df['MACD'], label='MACD', color='blue')
        plt.plot(self.df['MACD_signal'], label='Signal Line', color='orange')
        plt.title("📉 MACD and Signal Line")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summary_metrics(self):
        """
        Compute summary metrics for closing price and volume.

        Returns:
            dict: A dictionary of financial summary statistics.
        """
        summary = {
            'Average Close': self.df['Close'].mean(),
            'Max Close': self.df['Close'].max(),
            'Min Close': self.df['Close'].min(),
            'Volume Std Dev': self.df['Volume'].std()
        }
        return summary

    def show_summary(self):
        """
        Print summary metrics in a readable format.
        """
        metrics = self.summary_metrics()
        print("\n📊 Financial Summary:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")

    def fetch_from_yfinance(self, symbol='AAPL', period='6mo', interval='1d') -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance and assign to self.df

        Args:
            symbol (str): Ticker symbol (e.g. 'AAPL')
            period (str): Time period (e.g. '6mo', '1y')
            interval (str): Data interval (e.g. '1d')

        Returns:
            pd.DataFrame: The fetched DataFrame
        """
        try:
            print(f"\n📈 Downloading {symbol} stock data using yfinance...")
            df = yf.download(symbol, period=period, interval=interval)
            df.reset_index(inplace=True)
            df['Company'] = symbol.upper()
            df.rename(columns={'Date': 'date'}, inplace=True)
            self.df = df
            print(df.head())
            return df
        except Exception as e:
            print(f"❌ Error fetching data from yfinance: {e}")
            self.df = pd.DataFrame()
            return pd.DataFrame()

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Calculate basic technical indicators for the stock data.

        Args:
            data (pd.DataFrame): Historical stock data

        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        df = self.df.copy()

        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        self.df = df
        return df

    def plot_stock_data(self, indicators: bool = True, volume: bool = True) -> None:
        """
        Create an interactive stock price chart with candlesticks, indicators, and volume.

        Args:
            indicators (bool): If True, plot MA20 and MA50.
            volume (bool): If True, plot volume bars.
        """
        fig = make_subplots(
            rows=2 if volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3] if volume else [1]
        )

        fig.add_trace(
            go.Candlestick(
                x=self.df['date'],
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        if indicators:
            if 'MA20' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df['date'],
                        y=self.df['MA20'],
                        name='MA 20',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
            if 'MA50' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df['date'],
                        y=self.df['MA50'],
                        name='MA 50',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )

        if volume:
            fig.add_trace(
                go.Bar(
                    x=self.df['date'],
                    y=self.df['Volume'],
                    name='Volume'
                ),
                row=2, col=1
            )

        fig.update_layout(
            title="Stock Price Overview",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )

        fig.show()

