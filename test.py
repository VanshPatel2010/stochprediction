import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.title("Stock Market Prediction using Machine Learning")

# User inputs
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL")
period = st.selectbox("Select Data Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=4)
window = st.slider("Select Window Size (days):", min_value=3, max_value=30, value=5)

if st.button("Predict"):
    with st.spinner("Fetching and processing data..."):
        # Download data
        df = yf.download(symbol, period=period)

        if df.empty:
            st.error(f"No data available for symbol {symbol} over the period {period}.")
        else:
            # Feature engineering
            window_size = window
            df['SMA'] = df['Close'].rolling(window=window_size).mean()

            # Prepare data
            X = []
            y = []
            for i in range(window_size, len(df)):
                X.append(df['Close'].iloc[i-window_size:i].values)
                y.append(df['Close'].iloc[i])

            X = np.array(X)
            y = np.array(y)
            X = X.reshape(X.shape[0], -1)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")

            # Plot results
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index[-len(y_test):], y_test, label="Actual Price", color="blue")
            ax.plot(df.index[-len(y_test):], y_pred, label="Predicted Price", color="red")
            ax.set_title(f"{symbol} Stock Price Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)

            # Display predictions
            st.write("Predicted vs Actual Prices:")
            results = pd.DataFrame({"Predicted": y_pred, "Actual": y_test})
            st.write(results)

