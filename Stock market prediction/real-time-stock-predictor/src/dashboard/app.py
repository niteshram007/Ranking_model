import os
import streamlit as st
import requests
import time
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD


st.set_page_config(page_title='Real-Time Stock Predictor', layout='wide')
st.title("Real-Time Stock Predictor")

ticker = st.sidebar.text_input("Ticker", "AAPL")
seq_len = st.sidebar.number_input("Sequence length", min_value=10, max_value=240, value=60)
poll = st.sidebar.checkbox("Live polling", value=False)
interval = st.sidebar.number_input("Polling interval (s)", min_value=1, max_value=60, value=5)

col1, col2 = st.columns([3, 1])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError('Close column required')
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    try:
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
    except Exception:
        df['MACD'] = 0.0
    try:
        df['RSI'] = RSIIndicator(df['Close']).rsi()
    except Exception:
        df['RSI'] = 0.0
    if 'Volume' in df.columns:
        df['vol_10'] = df['Volume'].rolling(10).std()
    else:
        df['vol_10'] = 0.0
    return df.fillna(method='bfill').fillna(method='ffill')


def fetch_recent_rows(ticker: str, rows: int):
    # try synthetic CSV first
    path = f'data/{ticker}_synthetic.csv'
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        # fallback to yfinance (daily)
        try:
            df = yf.download(ticker, period='90d', interval='1d', progress=False)
        except Exception:
            raise RuntimeError('Could not fetch data')
    df = add_indicators(df)
    features = ['Close', 'MA10', 'MA50', 'MACD', 'RSI', 'vol_10']
    for c in features:
        if c not in df.columns:
            df[c] = 0.0
    last = df[features].tail(rows).values.tolist()
    return df, last


with col2:
    if st.button('Ping server'):
        try:
            r = requests.get('http://127.0.0.1:8003/ping', timeout=2)
            st.success('Server OK')
        except Exception as e:
            st.error(f'Cannot reach server: {e}')

with col1:
    st.subheader(f'Live prediction for {ticker}')
    chart_placeholder = st.empty()
    metric_placeholder = st.empty()

    def do_prediction_once():
        try:
            df, rows = fetch_recent_rows(ticker, seq_len)
        except Exception as e:
            st.error(f'Error fetching data: {e}')
            return
        try:
            r = requests.post('http://127.0.0.1:8003/predict_data', json={'ticker': ticker, 'data': rows}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                metric_placeholder.metric(label=f'Predicted next price for {ticker}', value=f"${data['prediction']:.2f}")
            else:
                metric_placeholder.error(f'Predict error: {r.text}')
        except Exception as e:
            metric_placeholder.error(f'Error calling server: {e}')

        # plot last closes
        try:
            last_plot = df.tail(200)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=last_plot.index, y=last_plot['Close'], name='Close'))
            if 'MA10' in last_plot.columns:
                fig.add_trace(go.Scatter(x=last_plot.index, y=last_plot['MA10'], name='MA10'))
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    if st.button('Get Prediction'):
        do_prediction_once()

    if poll:
        count = st.sidebar.number_input('Poll iterations', min_value=1, max_value=1000, value=60)
        stop = st.sidebar.button('Stop polling')
        for i in range(count):
            do_prediction_once()
            time.sleep(interval)
            if stop:
                break


st.markdown('---')
st.header('Top picks – Indian stocks and mutual funds')


def _parse_tickers(s: str):
    return [t.strip() for t in s.split(',') if t.strip()]


default_stocks = 'RELIANCE.NS,TCS.NS,HDFCBANK.NS'
default_mfs = ''

stocks_input = st.text_input('Indian stock tickers (.NS, comma separated)', default_stocks)
mfs_input = st.text_input('Mutual fund tickers (comma separated, if available on Yahoo Finance)', default_mfs)

stock_tickers = _parse_tickers(stocks_input)
mf_tickers = _parse_tickers(mfs_input)

API_BASE = os.getenv('BACKEND_API_BASE_URL', 'http://127.0.0.1:8003')


if st.button('Find Top Picks'):
    tickers = stock_tickers + mf_tickers
    categories = ['Stock'] * len(stock_tickers) + ['Mutual Fund'] * len(mf_tickers)

    if not tickers:
        st.info('Please enter at least one ticker.')
    else:
        try:
            payload = {
                'tickers': tickers,
                'categories': categories,
                'top_k': len(tickers),
            }
            r = requests.post(f'{API_BASE}/top_picks', json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if data:
                    df_rank = pd.DataFrame(data)
                    st.subheader('Ranked top picks (model-based outperformance probability)')
                    st.dataframe(df_rank)
                else:
                    st.info('No ranked results returned for the provided tickers.')
            else:
                st.error(f'Error from server: {r.status_code} {r.text}')
        except Exception as e:
            st.error(f'Error calling /top_picks API: {e}')