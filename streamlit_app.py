import streamlit as st
import pandas as pd
from run_forecast import forecast_stock
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Forecaster", layout="wide")
st.title("📈 Stock Price Forecasting Tool")

# --- Sidebar Inputs ---
st.sidebar.header("Forecast Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="TSLA")
lookback_years = st.sidebar.slider("Lookback Period (Years)", min_value=1, max_value=10, value=3)
predict_days = st.sidebar.slider("Days to Forecast", min_value=30, max_value=365, value=180)

# --- Run Forecast ---
if st.sidebar.button("Run Forecast"):
    with st.spinner("Downloading data and running forecast..."):
        hist_df, forecast_df = forecast_stock(ticker, lookback_years, predict_days)

    if hist_df is None:
        st.error("No data found. Please check the ticker symbol.")
    else:
        # Combine recent history and forecast
        display_hist = hist_df.tail(90)
        combined_df = pd.concat([display_hist, forecast_df[["Smoothed"]]], axis=0)

        # Plot
        st.subheader(f"{ticker.upper()} Forecast")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(display_hist.index, display_hist["Close"], label="Historical (Last 90 Days)", color="black")
        ax.plot(forecast_df.index, forecast_df["Smoothed"], label="Forecast (Smoothed)", color="orange", linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title(f"{ticker.upper()} Forecasted Daily Close")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Show data
        with st.expander("📄 See Forecast Data"):
            st.dataframe(forecast_df.style.format({"Predicted_Close": "{:.2f}", "Smoothed": "{:.2f}"}))
