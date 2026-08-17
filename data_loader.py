"""
data_loader.py
--------------
Data ingestion for NIFTY 50 index and its constituents from Yahoo Finance.

The module pulls daily OHLCV data, computes log-returns, and produces aligned
data frames for use in the HMM, GARCH, and HRP allocation layers.

All sources are public and reproducible.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# --- NIFTY 50 constituents (NSE tickers, suffix .NS on Yahoo) ---------------
# Composition as of 2024. A few names have changed over the 20-year window;
# we use the current composition as a tractable proxy. The HMM is fit on the
# NIFTY 50 index itself, not on the constituents, so this only affects HRP.
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "KOTAKBANK.NS", "HCLTECH.NS", "MARUTI.NS", "AXISBANK.NS",
    "SUNPHARMA.NS", "M&M.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "ASIANPAINT.NS", "ONGC.NS", "POWERGRID.NS", "WIPRO.NS", "ADANIENT.NS",
    "JSWSTEEL.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "COALINDIA.NS", "NESTLEIND.NS",
    "TATASTEEL.NS", "ADANIPORTS.NS", "GRASIM.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS",
    "BEL.NS", "TECHM.NS", "DRREDDY.NS", "CIPLA.NS", "INDUSINDBK.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "DIVISLAB.NS",
    "TATACONSUM.NS", "SHRIRAMFIN.NS", "SBILIFE.NS", "TRENT.NS", "HDFCLIFE.NS",
]

NIFTY50_INDEX = "^NSEI"  # NIFTY 50 index ticker on Yahoo


def fetch_index(
    ticker: str = NIFTY50_INDEX,
    start: str = "2004-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a single index/stock and compute log returns.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume, log_ret
    indexed by trading day.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    # yfinance returns MultiIndex columns for single ticker in some versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["log_ret"])
    return df


def fetch_constituents(
    tickers: list[str] = None,
    start: str = "2004-01-01",
    end: str = "2024-12-31",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers and return a wide
    DataFrame of log returns (rows: dates, cols: tickers).

    Tickers that fail to download or have insufficient history are dropped.
    If cache_path is given, results are cached as parquet for re-runs.
    """
    if tickers is None:
        tickers = NIFTY50_TICKERS

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    # yfinance with multiple tickers returns MultiIndex columns:
    # level 0 = price field (Open/High/Low/Close/Volume), level 1 = ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop tickers with too little data (need at least 5 years of history)
    min_obs = 252 * 5
    prices = prices.loc[:, prices.notna().sum() >= min_obs]

    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        log_returns.to_parquet(cache_path)

    return log_returns


def align_data(
    index_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align the NIFTY index and constituent return frames to a common date index.
    Returns (index_aligned, constituents_aligned).
    """
    common_idx = index_df.index.intersection(constituents_df.index)
    return index_df.loc[common_idx], constituents_df.loc[common_idx]
