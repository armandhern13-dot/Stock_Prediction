import numpy as np
import pandas as pd
from datetime import date, timedelta
import yfinance as yf
import pandas_datareader.data as web
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _make_session(timeout=120, retries=5):
    """
    Creates a requests.Session with retries + a higher timeout.
    This helps prevent ReadTimeoutError when calling FRED.
    """
    s = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    old_request = s.request

    def request(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return old_request(method, url, **kwargs)

    s.request = request
    return s


def extract_features():
    """
    Builds the feature matrix X used by the model.

    Target in notebook example: UBER_Future
    Features returned by this function (X):
      SBUX, NKE, CMG, DEXJPUS, DEXUSUK, SP500, DJIA, VIXCLS
    """
    return_period = 5

    # Keep this smaller to reduce FRED timeouts
    END_DATE = date.today()
    START_DATE = END_DATE - timedelta(days=365)  # 1 year

    START_DATE = START_DATE.strftime("%Y-%m-%d")
    END_DATE = END_DATE.strftime("%Y-%m-%d")

    # Stocks: include UBER because you may need it for the target in training,
    # but features will be SBUX/NKE/CMG.
    stk_tickers = ["UBER", "SBUX", "NKE", "CMG"]

    ccy_tickers = ["DEXJPUS", "DEXUSUK"]
    idx_tickers = ["SP500", "DJIA", "VIXCLS"]

    # Download stock data
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)

    # Build a robust session for FRED
    session = _make_session(timeout=120, retries=5)

    # Fetch FRED series one-by-one (more reliable than list-at-once)
    def fred_series(series_list):
        frames = []
        for s in series_list:
            df = web.DataReader(s, "fred", start=START_DATE, end=END_DATE, session=session)
            frames.append(df)
        return pd.concat(frames, axis=1)

    ccy_data = fred_series(ccy_tickers)
    idx_data = fred_series(idx_tickers)

    # ----- Target (Y) used in notebook training (optional here) -----
    # If you only want features for inference, you can skip computing Y,
    # but keeping it matches your original structure.
    Y = np.log(stk_data.loc[:, ("Adj Close", "UBER")]).diff(return_period).shift(-return_period)
    Y.name = "UBER_Future"

    # ----- Features (X) -----
    # Stock features: SBUX, NKE, CMG
    X1 = np.log(stk_data.loc[:, ("Adj Close", ["SBUX", "NKE", "CMG"])]).diff(return_period)
    X1.columns = X1.columns.droplevel(0)

    # Macro features
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)

    # Align and clean
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    dataset.index.name = "Date"

    # Return ONLY the feature columns (X) in the correct order
    features = dataset[["SBUX", "NKE", "CMG", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"]]
    features = features.sort_index().reset_index(drop=True)

    return features


def get_bitcoin_historical_prices(days=60):
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    # Add timeout + basic error handling
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["Timestamp", "Close Price (USD)"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms").dt.normalize()
    df = df[["Date", "Close Price (USD)"]].set_index("Date")
    return df


