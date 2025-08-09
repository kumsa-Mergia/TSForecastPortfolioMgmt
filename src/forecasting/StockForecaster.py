import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class StockForecaster:
    def __init__(self, ticker, start="2015-01-01", end=None):
        self.ticker = ticker
        self.start = start
        self.end = end or datetime.today().strftime("%Y-%m-%d")
        self.data = None
        self.prices = None
        self.returns = None
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.arima_model_fit = None

    def fetch_data(self):
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start, end=self.end)
        self.prices = self.data['Close'].fillna(method='ffill')
        self.returns = self.prices.pct_change().dropna()


    def plot_price_history(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.prices, label=f'{self.ticker} Price')
        plt.title(f'{self.ticker} Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def check_stationarity(self, series):
        result = adfuller(series.dropna())
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        if result[1] <= 0.05:
            print("Series is stationary")
        else:
            print("Series is non-stationary")

    def decompose_series(self):
        decomposition = seasonal_decompose(self.prices, model='multiplicative', period=252)
        decomposition.plot()
        plt.show()


    def forecast_arima(self, order=(5, 1, 2), train_ratio=0.9):
        print("\nTraining ARIMA model...")
        train_size = int(len(self.prices) * train_ratio)
        train, test = self.prices[:train_size], self.prices[train_size:]

        model = ARIMA(train, order=order)
        self.arima_model_fit = model.fit()

        forecast = self.arima_model_fit.forecast(steps=len(test))
        self.arima_forecast_results = (test.index, test, forecast)

    def plot_arima_forecast(self):
        if self.arima_forecast_results is None:
            print("No ARIMA forecast available. Run forecast_arima() first.")
            return

        test_idx, test, forecast = self.arima_forecast_results
        plt.figure(figsize=(14,6))
        plt.plot(self.prices[-len(test):], label='Historical Price')
        plt.plot(test_idx, forecast, label='ARIMA Forecast', linestyle='--')
        plt.title(f"{self.ticker} ARIMA Forecast vs Actual")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def forecast_lstm(self, window=60, epochs=20, batch_size=32):
        print("\nTraining LSTM model...")
        scaled_data = self.scaler.fit_transform(self.prices.values.reshape(-1, 1))

        def create_sequences(data, window):
            X, y = [], []
            for i in range(len(data) - window):
                X.append(data[i:i+window])
                y.append(data[i+window])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, window)
        X_train, X_test = X, X[-window:]
        y_train, y_test = y, y[-window:]

        self.lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(window, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1)
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predict on test data
        lstm_forecast = self.lstm_model.predict(X_test)
        lstm_forecast = self.scaler.inverse_transform(lstm_forecast).flatten()
        self.lstm_forecast_results = (self.prices.index[-window:], self.prices[-window:], lstm_forecast)

    def plot_lstm_forecast(self):
        if self.lstm_forecast_results is None:
            print("No LSTM forecast available. Run forecast_lstm() first.")
            return

        idx, actual, forecast = self.lstm_forecast_results
        plt.figure(figsize=(14,6))
        plt.plot(idx, actual, label='Historical Price')
        plt.plot(idx, forecast, label='LSTM Forecast', linestyle='--')
        plt.title(f"{self.ticker} LSTM Forecast vs Actual")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    # The combined 1-year forecast method from before (optional)
    def forecast_next_year(self, window=60):
        future_steps = 252  # 1 year trading days

        # ARIMA forecast
        arima_forecast = self.arima_model_fit.forecast(steps=future_steps)

        # LSTM forecast
        lstm_input = self.scaler.transform(self.prices.values.reshape(-1, 1))
        lstm_sequence = lstm_input[-window:]
        lstm_forecast = []

        for _ in range(future_steps):
            pred = self.lstm_model.predict(lstm_sequence.reshape(1, window, 1), verbose=0)
            lstm_forecast.append(pred[0][0])
            lstm_sequence = np.append(lstm_sequence[1:], pred, axis=0)

        lstm_forecast = self.scaler.inverse_transform(np.array(lstm_forecast).reshape(-1,1)).flatten()

        future_dates = pd.date_range(
            start=self.prices.index[-1],
            periods=future_steps+1,
            inclusive='right'
            )


        plt.figure(figsize=(14,6))
        plt.plot(self.prices[-252:], label='Historical (Last Year)', color='blue')
        plt.plot(future_dates, arima_forecast, label='ARIMA Forecast (1Y)', linestyle='--', color='red')
        plt.plot(future_dates, lstm_forecast, label='LSTM Forecast (1Y)', linestyle='--', color='green')
        plt.title(f"{self.ticker} 1-Year Forecast (ARIMA vs LSTM)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()