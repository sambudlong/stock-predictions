from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import yfinance as yf

def forecast_stock(ticker, lookback_years=3, forecast_days=30):

    # Download and Prepare Data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_years * 365)

    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()

    # Feature Engineering
    def add_features(data):
        df = data.copy()
        df["Return_1"] = df["Close"].pct_change()
        for window in [2, 5, 10, 30, 60, 90, 180, 250]:
            df[f"MA_{window}"] = df["Close"].rolling(window).mean()
            df[f"STD_{window}"] = df["Close"].rolling(window).std()
        df = df.dropna()
        return df

    features_df = add_features(df)

    # Train-Test Split
    # train = features_df.loc[:'2024-12-31']
    train = features_df.loc[:end_date]
    X_train = train.drop(columns=["Close"])
    y_train = train["Close"]
    used_features = X_train.columns.tolist()

    # Train Multiple Models
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, random_state=42)

    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # Forecast 2025 Daily Prices
    last_real_date = features_df.index[-1]
    last_real_price = df["Close"].iloc[-1]

    forecast_start = end_date + timedelta(days=1)
    future_dates = pd.date_range(start=forecast_start, periods=forecast_days, freq='B')
    # forecast_start = datetime(2025, 1, 1)
    # forecast_end = datetime(2025, 8, 1)

    synthetic = df.copy()
    predictions = []

    for date in future_dates:
        latest = add_features(synthetic)
        try:
            X_pred = latest[used_features].iloc[[-1]]
        except IndexError:
            print(f"Skipping {date}: not enough history.")
            continue
        if X_pred.isnull().any().any():
            print(f"Skipping {date}: NaNs in input.")
            continue

        # Base predictions
        pred_ridge = ridge.predict(X_pred)[0]
        pred_rf = rf.predict(X_pred)[0]
        pred_xgb = xgb.predict(X_pred)[0]
        base_pred = np.mean([pred_ridge, pred_rf, pred_xgb])

        # Inject general upward trend (0.1% daily growth compounded)
        trend_factor = 1.005
        trended_pred = base_pred * trend_factor

        # Add volatility (recent 30-day rolling std dev)
        recent_std = latest["STD_250"].iloc[-1] if "STD_250" in latest.columns else 5
        noise = np.random.normal(loc=0, scale=recent_std)
        final_pred = trended_pred + noise

        # Ensure the price doesn't drop below a threshold (optional)
        final_pred = max(final_pred, 0.01)

        synthetic = pd.concat([synthetic, pd.DataFrame({"Close": [final_pred]}, index=[date])])
        predictions.append((date, final_pred))


    # Plot Results
    predicted_df = pd.DataFrame(predictions, columns=["Date", "Predicted_Close"]).set_index("Date")
    predicted_df = pd.concat([
        pd.DataFrame({"Predicted_Close": [last_real_price]}, index=[last_real_date]),
        predicted_df
    ])
    plt.figure(figsize=(14, 6))
    plt.plot(df["Close"], label="Historical Close")
    plt.plot(predicted_df["Predicted_Close"], label="Predicted 2025 Close", color="orange")
    plt.title(f"{ticker} Close Price Prediction for 2025\n(Upward Trend + Volatility + Model Ensemble)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
