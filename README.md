# Predicting Price Moves with News Sentiment

This project is part of the **10 Academy AIM Week 1 Challenge**. The goal is to assess the relationship between financial news sentiment and corresponding stock market movements using a combination of **NLP**, **time series analysis**, and **quantitative finance techniques**.


## Challenge Overview
The Week 1 challenge focuses on building a pipeline that integrates:
- Financial news headlines
- Daily stock prices
- Sentiment scoring
- Technical indicators
- Statistical correlation

Each task builds upon the previous one to form an end-to-end market signal analysis tool.


## Business Objective
The purpose of this analysis is to determine whether daily sentiment in financial news headlines can be used as a signal to predict or understand daily stock price movements. This has practical applications for:
- Building algorithmic trading strategies
- Enhancing financial forecasting models
- Automating news-based market reactions


## Dataset Overview
### News Dataset
- **Source:** Raw analyst ratings and headlines
- **File:** [`data/raw_analyst_ratings.csv`](https://drive.google.com/drive/folders/1rsispvTGPjC8pbKS-yYb-6dcJiXTKSAv?usp=drive_link)
- **Key Columns:**
  - `headline`: Title of the article
  - `publisher`: Source of news
  - `date`: Date of publication (UTC-4)
  - `stock`: Associated stock ticker

### Stock Dataset
- **Source:** Yahoo Finance (`yfinance` API)
- **File:** `data/extracted/combined_stocks.csv`
- **Key Columns:**
  - `Date`: Trading day
  - `Open`, `High`, `Low`, `Close`, `Volume`: Daily OHLCV data


## Repository Structure
```
├── .github/
│   └── workflows/
│       └── ci.yml                   # Continuous integration workflow
│
├── data/
│   ├── extracted/
│   │   └── combined_stocks.csv      # Cleaned and merged stock data
│   └── raw_analyst_ratings.csv      # Raw news headlines data
│
├── notebooks/
│   ├── correlation_analysis.ipynb   # Task 3: Sentiment vs stock return
│   ├── exploratory_EDA.ipynb        # Task 1: EDA on news and stock data
│   └── technical_analysis.ipynb     # Task 2: Technical indicators analysis
│
├── scripts/                         # Placeholder for script utilities (if needed)
│
├── src/                             # Modular Python scripts
│   ├── correlation_analysis.py      # Stock returns & correlation metrics
│   ├── data_alignment.py            # Normalize and align datasets by date
│   ├── descriptive_stats.py         # Headline length & publisher stats
│   ├── ingest_data.py               # Load and clean data from CSVs
│   ├── sentiment_analysis.py        # Sentiment scoring and aggregation
│   ├── technical_analysis.py        # Stock indicators (RSI, MACD, MA)
│   ├── text_analysis.py             # Topic modeling, keyword extraction
│   └── time_series_analysis.py      # Frequency and timestamp trends
│
├── tests/                           # Unit and integration tests (TBD)
│
├── requirements.txt
└── README.md
```

---

## Task Implementation Summary

### Task 1: Environment Setup & EDA
- GitHub repo initialized with CI/CD pipeline
- `.gitignore`, `README`, and `requirements.txt` created
- Performed descriptive analysis:
  - Headline lengths
  - Most frequent publishers
  - News frequency over time

### Task 2: Technical Indicator Analysis
- Downloaded historical stock data using `yfinance`
- Computed technical indicators using `TA-Lib`:
  - MA20, MA50, RSI, MACD
- Built interactive candlestick and line charts with Plotly
- Encapsulated in `technical_analysis.py`

### Task 3: News-Sentiment Correlation
- Parsed and aligned `date` fields across datasets
- Applied sentiment scoring using `TextBlob`
- Aggregated sentiment by date
- Calculated stock returns using closing prices
- Merged datasets and computed Pearson correlation:
  > **r = 0.0013**, **p = 0.8729** (not statistically significant)


## Final Analysis: Sentiment vs Stock Return
> **Pearson correlation = 0.0013, p-value = 0.8729**

### Interpretation
- No linear relationship found between same-day sentiment and returns
- High p-value confirms lack of statistical significance
- Suggests news sentiment alone is not a reliable same-day signal


## Key Visuals and Graphs (in notebooks)
- Line plots of stock closing prices
- RSI and MACD indicators
- Scatter plot of sentiment vs daily return
- Histograms of headline lengths and GHI distribution


## How to Run
```bash
# Clone the repository
$ git clone https://github.com/michaWorku/news-sentiment-stock-analysis.git
$ cd news-sentiment-stock-analysis

# Create environment and install dependencies
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

# Run any notebook
$ jupyter notebook notebooks/correlation_analysis.ipynb
```

## Learning Outcomes
- Data ingestion, preprocessing, and alignment
- Exploratory data analysis and visualization
- Sentiment analysis using NLP
- Financial indicators and return modeling
- Correlation analysis with statistical inference


## Contributors
- Mikias Worku

## 📜 License
This project is licensed under the MIT License.
