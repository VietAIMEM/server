from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("viet_model.keras")
with open("viet_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("trained_tickers.pkl", "rb") as f:
    trained_tickers = pickle.load(f)

def get_latest_61_minutes(ticker, period="5d", interval="1m"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df = df[["Close"]].reset_index()
    df = df.rename(columns={"Close": "close", "Datetime": "datetime"})
    df["ticker"] = ticker
    if len(df) < 61:
        raise ValueError(f"Not enough data for {ticker}. Only {len(df)} records available.")
    df = df.tail(61)
    return df

def preprocess_for_prediction_new_scaler(df, ticker, label_encoder, window_size=60):
    if ticker not in label_encoder.classes_:
        raise ValueError(f"Ticker {ticker} not in LabelEncoder classes.")
    ticker_id = label_encoder.transform([ticker])[0]
    scaler = StandardScaler()
    close_values = df["close"].values
    if len(close_values) != window_size + 1:
        raise ValueError(f"Expected {window_size + 1} records, got {len(close_values)}.")
    scaler.fit(close_values[:window_size].reshape(-1, 1))
    normalized_values = scaler.transform(close_values.reshape(-1, 1)).flatten()
    X_price = normalized_values[:window_size].reshape(1, window_size, 1)
    X_ticker = np.array([[ticker_id]])
    y_true = normalized_values[window_size]
    return X_price, X_ticker, y_true, scaler

@app.get("/tickers")
async def get_tickers():
    return {"tickers": trained_tickers}

@app.get("/predict/{ticker}")
async def predict_ticker(ticker: str):
    ticker = ticker.upper()
    if ticker not in trained_tickers:
        return {"error": f"Ticker {ticker} chưa được huấn luyện."}
    try:
        df = get_latest_61_minutes(ticker)
        X_price, X_ticker, y_true, scaler = preprocess_for_prediction_new_scaler(df, ticker, label_encoder)

        pred_normalized = model.predict({"price_input": X_price, "ticker_input": X_ticker}, verbose=0)
        pred_original = scaler.inverse_transform(pred_normalized)[0][0]
        y_true_original = scaler.inverse_transform([[y_true]])[0][0]

        pred_original = float(pred_original)
        y_true_original = float(y_true_original)
        error_value = float(abs(pred_original - y_true_original))

        return {
            "ticker": ticker,
            "predicted_price": round(pred_original, 4),
            "actual_price": round(y_true_original, 4),
            "error": round(error_value, 4)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/history/{ticker}")
async def get_history(ticker: str):
    ticker = ticker.upper()
    if ticker not in trained_tickers:
        return {"error": f"Ticker {ticker} chưa được huấn luyện."}
    try:
        df = get_latest_61_minutes(ticker)
        history = df[["datetime", "close"]].to_dict('records')
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}