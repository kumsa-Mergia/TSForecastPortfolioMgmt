import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import os

plt.style.use("seaborn-v0_8")
sns.set_theme()

class FinancialDataAnalyzer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
    
    def download_data(self):
        print("Downloading data...")
        for ticker in self.tickers:
            # Set auto_adjust=False to keep Adj Close column
            df = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
            self.data[ticker] = df
            print(f"Downloaded {ticker} with {df.shape[0]} rows")
    
    def clean_data(self):
        for ticker, df in self.data.items():
            print(f"Cleaning data for {ticker}...")
            df.interpolate(method='time', inplace=True)
            df.ffill(inplace=True)
            self.data[ticker] = df
    
    def feature_engineering(self):
        for ticker, df in self.data.items():
            df['Daily_Return'] = df['Adj Close'].pct_change()
            df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=21).std()
            df['Rolling_Mean'] = df['Adj Close'].rolling(window=21).mean()
    
    def plot_price_trends(self):
        plt.figure(figsize=(14, 7))
        for ticker, df in self.data.items():
            plt.plot(df.index, df['Adj Close'], label=ticker)
        plt.title('Adjusted Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.show()
    
    def plot_volatility(self, ticker):
        if ticker not in self.data:
            print(f"No data found for {ticker}")
            return
        df = self.data[ticker]
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['Rolling_Volatility'], label=f'{ticker} Rolling Volatility (21 days)')
        plt.title(f'{ticker} Rolling Volatility (21 days)')
        plt.show()
    
    def adf_test(self, series, title=''):
        print(f'\nADF Test: {title}')
        result = adfuller(series.dropna())
        labels = ['ADF Statistic', 'p-value', '# Lags Used', 'Number of Observations Used']
        for value, label in zip(result[:4], labels):
            print(f'{label}: {value}')
        if result[1] <= 0.05:
            print("=> Series is stationary")
        else:
            print("=> Series is non-stationary")
    
    def run_stationarity_tests(self):
        for ticker, df in self.data.items():
            print(f"\nTesting stationarity for {ticker}:")
            self.adf_test(df['Adj Close'], f'{ticker} Adjusted Close')
            self.adf_test(df['Daily_Return'], f'{ticker} Daily Returns')
    
    def calculate_var_sharpe(self, ticker, confidence_level=0.05):
        if ticker not in self.data:
            print(f"No data for {ticker}")
            return None, None
        
        returns = self.data[ticker]['Daily_Return'].dropna()
        var = np.percentile(returns, 100 * confidence_level)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        print(f"{ticker} VaR at {confidence_level*100}%: {var:.4f}")
        print(f"{ticker} Annualized Sharpe Ratio: {sharpe_ratio:.4f}")
        return var, sharpe_ratio
    def save_raw_data(self, save_dir='../data/processed'):
        # Make sure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        for ticker, df in self.data.items():
            file_path = os.path.join(save_dir, f"{ticker}_processed.csv")
            df.to_csv(file_path)
            print(f"Saved {ticker} processed data to {file_path}")