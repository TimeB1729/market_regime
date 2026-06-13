"""
Regime profiling: compute per-regime performance statistics.

Metrics
-------
- Mean / Median daily log-return
- Volatility (daily std)
- Annualised volatility
- Annualised Sharpe ratio (rf = 0)
- Maximum drawdown
- Average regime duration (consecutive days)
- Frequency (% of observations)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR, REGIME_LABELS, REPORTS_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a returns series."""
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def _average_duration(labels: pd.Series, regime: str) -> float:
    """Average number of consecutive days spent in *regime*."""
    durations: list[int] = []
    count = 0
    for lbl in labels:
        if lbl == regime:
            count += 1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_regimes(
    labels: pd.Series,
    prices: pd.Series,
) -> pd.DataFrame:
    """Compute per-regime statistics.

    Parameters
    ----------
    labels : pd.Series  string regime labels, same index as prices
    prices : pd.Series  close prices

    Returns
    -------
    pd.DataFrame  one row per regime
    """
    returns = np.log(prices / prices.shift(1)).dropna()

    # Align
    shared  = labels.index.intersection(returns.index)
    labels  = labels.loc[shared]
    returns = returns.loc[shared]

    regimes = sorted(labels.unique())
    rows: list[dict] = []

    for regime in regimes:
        mask = labels == regime
        r    = returns.loc[mask]

        mean_ret  = float(r.mean())
        med_ret   = float(r.median())
        vol_daily = float(r.std())
        ann_vol   = vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
        ann_ret   = mean_ret  * TRADING_DAYS_PER_YEAR
        sharpe    = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
        mdd       = _max_drawdown(r)
        avg_dur   = _average_duration(labels, regime)
        freq      = mask.sum() / len(labels)

        rows.append({
            "Regime":               regime,
            "Mean Return (daily)":  mean_ret,
            "Median Return (daily)":med_ret,
            "Volatility (daily)":   vol_daily,
            "Ann. Volatility":      ann_vol,
            "Ann. Sharpe":          sharpe,
            "Max Drawdown":         mdd,
            "Avg Duration (days)":  avg_dur,
            "Frequency (%)":        freq * 100,
            "N Observations":       int(mask.sum()),
        })

    df = pd.DataFrame(rows).set_index("Regime")
    return df


def print_regime_report(profile: pd.DataFrame) -> None:
    """Print a formatted regime profile report to stdout."""
    print("\n" + "═" * 70)
    print("REGIME PROFILE REPORT")
    print("═" * 70)
    for regime, row in profile.iterrows():
        print(f"\n▶  {regime}")
        print(f"   Mean Return (daily)   : {row['Mean Return (daily)']:+.6f}")
        print(f"   Median Return (daily) : {row['Median Return (daily)']:+.6f}")
        print(f"   Volatility (daily)    :  {row['Volatility (daily)']:.6f}")
        print(f"   Ann. Volatility       :  {row['Ann. Volatility']:.4f}  ({row['Ann. Volatility']*100:.2f}%)")
        print(f"   Ann. Sharpe Ratio     :  {row['Ann. Sharpe']:.4f}")
        print(f"   Max Drawdown          :  {row['Max Drawdown']:.4f}  ({row['Max Drawdown']*100:.2f}%)")
        print(f"   Avg Duration (days)   :  {row['Avg Duration (days)']:.1f}")
        print(f"   Frequency             :  {row['Frequency (%)']:.2f}%  ({int(row['N Observations'])} obs)")
    print("\n" + "═" * 70)


def save_regime_report(profile: pd.DataFrame, split: str = "full") -> None:
    """Save regime profile to CSV."""
    path = REPORTS_DIR / f"regime_profile_{split}.csv"
    profile.to_csv(path)
    print(f"Regime profile saved → {path}")
