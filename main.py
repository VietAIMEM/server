from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from datetime import time, datetime, timedelta
import pytz

app = FastAPI()

# CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình và dữ liệu huấn luyện
model = load_model("viet_model.keras")
with open("viet_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("trained_tickers.pkl", "rb") as f:
    trained_tickers = pickle.load(f)

# ---------- Hàm xử lý ----------
def get_latest_5_days(ticker, interval="1m"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="7d", interval=interval)
    df = df.reset_index()
    df = df.rename(columns={"Close": "close", "Datetime": "datetime"})
    df["ticker"] = ticker
    return df

def is_in_trading_hours_utc(t: time):
    return time(13, 30) <= t <= time(19, 59)  # 9:30–15:59 NYT = 13:30–19:59 UTC

def preprocess_for_prediction_new_scaler(df, ticker, label_encoder, window_size=60):
    if ticker not in label_encoder.classes_:
        raise ValueError(f"Ticker {ticker} not in LabelEncoder classes.")
    ticker_id = label_encoder.transform([ticker])[0]
    scaler = StandardScaler()
    close_values = df["close"].values
    if len(close_values) < window_size:
        raise ValueError(f"Expected at least {window_size} records, got {len(close_values)}.")
    # scaler.fit(close_values.reshape(-1, 1))
    scaler.fit(close_values[-61:].reshape(-1, 1))
    normalized_values = scaler.transform(close_values[-window_size:].reshape(-1, 1)).flatten()
    X_price = normalized_values.reshape(1, window_size, 1)
    X_ticker = np.array([[ticker_id]])
    return X_price, X_ticker, scaler

@app.get("/tickers")
async def get_tickers():
    return {"tickers": trained_tickers}

@app.get("/history/{ticker}")
async def get_history(ticker: str):
    ticker = ticker.upper()
    if ticker not in trained_tickers:
        return {"error": f"Ticker {ticker} chưa được huấn luyện."}
    try:
        df = get_latest_5_days(ticker)
        df = df.dropna(subset=["close"])
        history = df[["datetime", "close"]].to_dict('records')
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict/{ticker}")
async def predict_ticker(ticker: str, type: str = "next_minute"):
    ticker = ticker.upper()
    if ticker not in trained_tickers:
        return {"error": f"Ticker {ticker} chưa được huấn luyện."}
    try:
        df = get_latest_5_days(ticker)
        df = df.dropna(subset=["close"])
        if type == "next_minute":
            window_size = 61
        elif type == "two_minutes":
            window_size = 62
        elif type == "3_minutes":
            window_size = 63
        elif type == "4_minutes":
            window_size = 64
        else:
            window_size = 65  # fallback


        df_last_n = df.tail(window_size)

        if len(df_last_n) < window_size:
            raise ValueError(f"Mã cổ phiếu này có vẻ đã không còn trên thị trường hoặc không có giao dịch trong thời gian gần đây. Vui lòng thử lại sau.")

        last_time = df_last_n["datetime"].iloc[-1].astimezone(pytz.UTC).time()
        next_day = not is_in_trading_hours_utc(last_time)

        X_price, X_ticker, scaler = preprocess_for_prediction_new_scaler(df_last_n, ticker, label_encoder)

        pred_normalized = model.predict({"price_input": X_price, "ticker_input": X_ticker}, verbose=0)
        pred_original = scaler.inverse_transform(pred_normalized)[0][0]
        actual_price = float(df_last_n["close"].iloc[-1])
        error_value = abs(pred_original - actual_price)

        response = {
            "ticker": ticker,
            "predicted_price": round(float(pred_original), 4),
            "actual_price": round(float(actual_price), 4),
            "error": round(float(error_value), 4),
            "next_day_prediction": next_day
        }

        if type == "two_minutes":
            df_extended = df_last_n.copy()
            df_extended.loc[df_last_n.index[-1] + 1, "close"] = pred_original
            df_extended = df_extended.tail(60)
            X_price_2, X_ticker_2, scaler_2 = preprocess_for_prediction_new_scaler(df_extended, ticker, label_encoder)
            pred_normalized_2 = model.predict({"price_input": X_price_2, "ticker_input": X_ticker_2}, verbose=0)
            pred_original_2 = scaler_2.inverse_transform(pred_normalized_2)[0][0]
            response["second_predicted_price"] = round(float(pred_original_2), 4)
        elif type in ["3_minutes", "4_minutes"]:
            minutes = 3 if type == "3_minutes" else 4


            df_extended = df_last_n.copy()
            predicted_prices = [pred_original]
            for _ in range(minutes - 1):
                df_extended.loc[df_extended.index[-1] + 1, "close"] = predicted_prices[-1]
                df_extended = df_extended.tail(60)
                X_price_n, X_ticker_n, scaler_n = preprocess_for_prediction_new_scaler(df_extended, ticker, label_encoder)
                pred_normalized_n = model.predict({"price_input": X_price_n, "ticker_input": X_ticker_n}, verbose=0)
                pred_original_n = scaler_n.inverse_transform(pred_normalized_n)[0][0]
                predicted_prices.append(pred_original_n)
            response["extended_predicted_prices"] = [round(float(p), 4) for p in predicted_prices]

        return response
    except Exception as e:
        return {"error": str(e)}
