"""
synthetic_data.py — Generate realistic SPY-like OHLCV data when Yahoo Finance
is unavailable.

Uses a four-state Markov-switching process calibrated to known SPY statistics:

  Bull:    μ = +0.0008/day,  σ = 0.008,  vol multiplier 1.0
  Bear:    μ = -0.0006/day,  σ = 0.010,  vol multiplier 1.2
  Sideways:μ = +0.0001/day,  σ = 0.006,  vol multiplier 0.8
  Crisis:  μ = -0.002/day,   σ = 0.025,  vol multiplier 3.0

Regime transition matrix is constructed to give realistic persistence
(avg bull ~120 days, crisis ~30 days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42

# ── Per-regime daily parameters ───────────────────────────────────────────────
REGIMES = {
    "Bull":     {"mu": 0.0008,  "sigma": 0.008},
    "Bear":     {"mu": -0.0006, "sigma": 0.010},
    "Sideways": {"mu": 0.0001,  "sigma": 0.006},
    "Crisis":   {"mu": -0.002,  "sigma": 0.025},
}

# Row-stochastic transition matrix  [Bull, Bear, Sideways, Crisis]
P = np.array([
    [0.990, 0.004, 0.004, 0.002],   # from Bull
    [0.005, 0.985, 0.007, 0.003],   # from Bear
    [0.008, 0.006, 0.982, 0.004],   # from Sideways
    [0.025, 0.025, 0.020, 0.930],   # from Crisis
])

STATE_NAMES = ["Bull", "Bear", "Sideways", "Crisis"]


def generate_spy_data(
    start: str = "2010-01-01",
    end:   str | None = None,
    seed:  int = RNG_SEED,
) -> pd.DataFrame:
    """Generate synthetic SPY OHLCV data.

    Parameters
    ----------
    start, end : ISO date strings
    seed       : random seed

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume]
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end or pd.Timestamp.today())
    n     = len(dates)

    # ── Simulate regime sequence via Markov chain ─────────────────────────────
    states = np.empty(n, dtype=int)
    states[0] = 0   # start in Bull
    for t in range(1, n):
        states[t] = rng.choice(4, p=P[states[t - 1]])

    # ── Simulate daily log-returns ────────────────────────────────────────────
    log_rets = np.empty(n)
    for t in range(n):
        params = REGIMES[STATE_NAMES[states[t]]]
        log_rets[t] = rng.normal(params["mu"], params["sigma"])

    # ── Build price series from log-returns ───────────────────────────────────
    prices = np.exp(np.cumsum(log_rets)) * 100.0   # start at 100

    # ── Build realistic OHLC ──────────────────────────────────────────────────
    intra_vol = np.abs(log_rets) * 0.5 + 0.002
    high   = prices * np.exp( rng.uniform(0, intra_vol))
    low    = prices * np.exp(-rng.uniform(0, intra_vol))
    open_  = prices * np.exp(rng.normal(0, 0.001, n))

    # Volume: higher in crisis / bear, auto-correlated
    base_vol = 80_000_000
    vol_mult = np.where(
        states == 3, rng.uniform(2.0, 4.0, n),   # crisis
        np.where(states == 1, rng.uniform(1.2, 2.0, n), 1.0)
    )
    volume = (base_vol * vol_mult * rng.uniform(0.8, 1.2, n)).astype(int)

    df = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  prices,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"

    print(f"[Synthetic] Generated {n} rows of SPY-like data "
          f"({dates[0].date()} → {dates[-1].date()})")
    return df
