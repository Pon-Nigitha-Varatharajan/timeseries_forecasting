import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fetch_stock_data(api_key, symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

def clean_and_preprocess_data(data):
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame(time_series).T.reset_index()
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df.sort_values(by="date", inplace=True)
    df.dropna(inplace=True)
    return df

def perform_eda(data):
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # Plot time series
    st.write("### Stock Price Over Time")
    plt.figure(figsize=(12, 6))
    plt.plot(data["date"], data["close"])
    plt.title("Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    st.pyplot(plt)
    
    # Heatmap of daily returns
    st.write("### Heatmap of Daily Returns")
    data["daily_return"] = data["close"].pct_change()
    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    heatmap_data = data.pivot_table(index="month", columns="day", values="daily_return", aggfunc="mean")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="viridis")
    plt.title("Heatmap of Daily Returns by Month and Day")
    st.pyplot(plt)

def check_stationarity(data):
    st.subheader("Stationarity Check")
    result = adfuller(data["close"])
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"p-value: {result[1]}")
    if result[1] > 0.05:
        st.write("The time series is **not stationary**.")
    else:
        st.write("The time series is **stationary**.")

def build_arima_model(data):
    st.subheader("Model Building and Evaluation")
    
    if len(data) < 2:
        st.error("Not enough data to build the ARIMA model. Please provide a larger dataset.")
        return
    

    train_size = int(len(data) * 0.8)
    train, test = data["close"][:train_size], data["close"][train_size:]
   
    if len(train) == 0 or len(test) == 0:
        st.error("Train or test set is empty. Please check the data splitting.")
        return
    

    try:
        model = ARIMA(train, order=(5, 1, 0))  
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=len(test))
        test = test.reset_index(drop=True)
        test_df = pd.DataFrame({"actual": test, "forecast": forecast})
        
        test_df.dropna(subset=["actual", "forecast"], inplace=True)
        
        if len(test_df) == 0:
            st.error("No valid data for evaluation after handling NaN values.")
            return
        
        st.write("### Forecast vs Actual")
        plt.figure(figsize=(12, 6))
        plt.plot(test_df.index, test_df["actual"], label="Actual")
        plt.plot(test_df.index, test_df["forecast"], label="Forecast")
        plt.title("Forecast vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Close Price (USD)")
        plt.legend()
        st.pyplot(plt)
        

        mae = mean_absolute_error(test_df["actual"], test_df["forecast"])
        rmse = np.sqrt(mean_squared_error(test_df["actual"], test_df["forecast"]))
        st.write(f"*MAE:* {mae}, *RMSE:* {rmse}")
    except Exception as e:
        st.error(f"Error building ARIMA model: {e}")


def main():
    st.title("Stock Price Forecasting with Alpha Vantage")
    

    api_key = st.text_input("Enter your Alpha Vantage API Key", type="password")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    
    if api_key and symbol:
        data = fetch_stock_data(api_key, symbol)
        if data is not None:
            cleaned_data = clean_and_preprocess_data(data)
            
            perform_eda(cleaned_data)
            
            check_stationarity(cleaned_data)
            
            build_arima_model(cleaned_data)

if __name__ == "__main__":
    main()