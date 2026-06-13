"""
Data loading and feature engineering for Market Regime Detection.

Features
--------
1. Log returns
2. 20-day rolling volatility
3. 20-day momentum
4. Volume z-score
5. Lag-1 autocorrelation of returns (rolling window)

All features are standardised (zero mean, unit variance) before model fitting.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from config import (
    TICKER,
    START_DATE,
    ROLLING_VOL_WINDOW,
    MOMENTUM_WINDOW,
    VOL_ZSCORE_WINDOW,
)

warnings.filterwarnings("ignore")


# ── Download ──────────────────────────────────────────────────────────────────

def download_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted OHLCV data from Yahoo Finance.

    Falls back to a synthetic SPY-like dataset when Yahoo Finance is
    unavailable (e.g. in network-restricted environments).

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPY".
    start  : str
        ISO-format start date.
    end    : str | None
        ISO-format end date; defaults to today.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Open, High, Low, Close, Volume].
    """
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("Empty result from yfinance")
        # yfinance ≥0.2 returns MultiIndex columns when auto_adjust=True
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw.index = pd.to_datetime(raw.index)
        raw.sort_index(inplace=True)
        print(f"Downloaded {len(raw)} rows for {ticker} "
              f"({raw.index[0].date()} → {raw.index[-1].date()})")
        return raw

    except Exception as exc:
        print(f"[WARNING] yfinance download failed ({exc}). "
              "Falling back to synthetic SPY data.")
        from synthetic_data import generate_spy_data
        return generate_spy_data(start=start, end=end)


# ── Feature engineering ───────────────────────────────────────────────────────

def _rolling_autocorr(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling lag-1 autocorrelation of *series*.

    Parameters
    ----------
    series : pd.Series
    window : int

    Returns
    -------
    pd.Series
    """
    def _autocorr1(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        s = pd.Series(x)
        return float(s.autocorr(lag=1))

    return series.rolling(window).apply(_autocorr1, raw=True)


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct and standardise features from raw OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data (must contain 'Close' and 'Volume').

    Returns
    -------
    features_scaled : pd.DataFrame
        Standardised feature matrix (NaN rows dropped).
    price_series    : pd.Series
        Close price series aligned with *features_scaled*.
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # 1. Log returns
    feat["log_return"] = np.log(close / close.shift(1))

    # 2. Rolling volatility (20-day)
    feat["rolling_vol"] = feat["log_return"].rolling(ROLLING_VOL_WINDOW).std()

    # 3. Momentum (20-day cumulative log return)
    feat["momentum"] = feat["log_return"].rolling(MOMENTUM_WINDOW).sum()

    # 4. Volume z-score (20-day)
    vol_mean = volume.rolling(VOL_ZSCORE_WINDOW).mean()
    vol_std  = volume.rolling(VOL_ZSCORE_WINDOW).std()
    feat["volume_zscore"] = (volume - vol_mean) / (vol_std + 1e-9)

    # 5. Lag-1 autocorrelation (20-day rolling)
    feat["autocorr_lag1"] = _rolling_autocorr(feat["log_return"], window=ROLLING_VOL_WINDOW)

    # Drop NaN rows (warm-up period)
    feat.dropna(inplace=True)
    price_aligned = close.loc[feat.index]

    # Standardise
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feat.values)
    features_scaled = pd.DataFrame(
        scaled_values,
        index=feat.index,
        columns=feat.columns,
    )

    print(f"Feature matrix: {features_scaled.shape[0]} rows × "
          f"{features_scaled.shape[1]} features "
          f"({features_scaled.index[0].date()} → {features_scaled.index[-1].date()})")

    return features_scaled, price_aligned


# ── Train / test split ────────────────────────────────────────────────────────

def temporal_split(
    features: pd.DataFrame,
    prices: pd.Series,
    train_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature matrix and price series at *train_end* (temporal only).

    Parameters
    ----------
    features  : pd.DataFrame
    prices    : pd.Series
    train_end : str  ISO date (inclusive)

    Returns
    -------
    train_features, test_features, train_prices, test_prices
    """
    mask = features.index <= pd.Timestamp(train_end)
    X_train = features.loc[mask]
    X_test  = features.loc[~mask]
    p_train = prices.loc[mask]
    p_test  = prices.loc[~mask]

    print(f"Train: {len(X_train)} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"Test : {len(X_test)} rows  ({X_test.index[0].date()} → {X_test.index[-1].date()})")
    return X_train, X_test, p_train, p_test
