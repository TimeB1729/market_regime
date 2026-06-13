"""
Changepoint detection from regime label sequences.

A "changepoint" is defined as a transition between two *persistent* regimes —
both the departing and arriving regime must sustain for at least
MIN_REGIME_DURATION trading days.

Additionally, a major changepoint must satisfy at least ONE of:

  (a) The previous-regime occupancy probability (posterior mean) exceeds
      OCCUPANCY_THRESHOLD.
  (b) The maximum posterior confidence on the transition day exceeds
      CONFIDENCE_THRESHOLD.

This two-layer filter removes intra-day noise and single-day regime flickers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from config import (
    MIN_REGIME_DURATION,
    OCCUPANCY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    REPORTS_DIR,
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RegimeRun:
    """A contiguous block of the same regime."""
    regime:   str
    start:    pd.Timestamp
    end:      pd.Timestamp
    duration: int


@dataclass
class Changepoint:
    """A meaningful structural transition between two persistent regimes."""
    date:          pd.Timestamp
    from_regime:   str
    to_regime:     str
    from_duration: int
    to_duration:   int
    confidence:    float = field(default=0.0)
    is_major:      bool  = field(default=False)


# ── Run-length encoding ───────────────────────────────────────────────────────

def encode_runs(labels: pd.Series) -> List[RegimeRun]:
    """Convert a label series into a list of contiguous RegimeRun blocks.

    Parameters
    ----------
    labels : pd.Series  string regime labels

    Returns
    -------
    List[RegimeRun]
    """
    runs: list[RegimeRun] = []
    if labels.empty:
        return runs

    current_regime = labels.iloc[0]
    start_idx      = labels.index[0]
    count          = 1

    for ts, lbl in zip(labels.index[1:], labels.iloc[1:]):
        if lbl == current_regime:
            count += 1
        else:
            runs.append(RegimeRun(
                regime=current_regime,
                start=start_idx,
                end=labels.index[labels.index.get_loc(ts) - 1],
                duration=count,
            ))
            current_regime = lbl
            start_idx      = ts
            count          = 1

    # Last run
    runs.append(RegimeRun(
        regime=current_regime,
        start=start_idx,
        end=labels.index[-1],
        duration=count,
    ))
    return runs


# ── Changepoint detection ─────────────────────────────────────────────────────

def detect_changepoints(
    labels:     pd.Series,
    posteriors: np.ndarray,
    min_duration: int = MIN_REGIME_DURATION,
) -> List[Changepoint]:
    """Detect all changepoints from a regime label sequence.

    A changepoint is recorded whenever two consecutive runs *both* have
    duration ≥ min_duration.

    Parameters
    ----------
    labels       : pd.Series   string regime labels
    posteriors   : np.ndarray  (n × n_components) posterior probabilities
    min_duration : int

    Returns
    -------
    List[Changepoint]
    """
    runs = encode_runs(labels)
    changepoints: list[Changepoint] = []

    for i in range(1, len(runs)):
        prev = runs[i - 1]
        curr = runs[i]

        if prev.duration < min_duration or curr.duration < min_duration:
            continue

        # Confidence = max posterior probability on the transition day
        trans_idx = labels.index.get_loc(curr.start)
        confidence = float(posteriors[trans_idx].max())

        cp = Changepoint(
            date=curr.start,
            from_regime=prev.regime,
            to_regime=curr.regime,
            from_duration=prev.duration,
            to_duration=curr.duration,
            confidence=confidence,
        )
        changepoints.append(cp)

    return changepoints


# ── Major changepoint filtering ───────────────────────────────────────────────

def filter_major_changepoints(
    changepoints: List[Changepoint],
    labels:       pd.Series,
    occupancy_threshold:  float = OCCUPANCY_THRESHOLD,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> List[Changepoint]:
    """Mark changepoints as *major* if they satisfy economic significance criteria.

    Criteria (OR logic):
    --------------------
    (a) The *from_regime* occupancy in the full label series > occupancy_threshold
    (b) Transition-day posterior confidence > confidence_threshold

    Parameters
    ----------
    changepoints         : list from detect_changepoints
    labels               : full label series (used for occupancy calculation)
    occupancy_threshold  : float  minimum regime frequency
    confidence_threshold : float  minimum day-level posterior confidence

    Returns
    -------
    List[Changepoint]  with is_major flag set appropriately
    """
    regime_freq: dict[str, float] = (
        labels.value_counts(normalize=True).to_dict()
    )

    major = []
    for cp in changepoints:
        occ = regime_freq.get(cp.from_regime, 0.0)
        crit_a = occ > occupancy_threshold
        crit_b = cp.confidence > confidence_threshold

        cp.is_major = crit_a or crit_b
        if cp.is_major:
            major.append(cp)

    return major


# ── Reporting ─────────────────────────────────────────────────────────────────

def changepoints_to_dataframe(changepoints: List[Changepoint]) -> pd.DataFrame:
    """Convert a list of Changepoint objects to a pandas DataFrame."""
    rows = [
        {
            "Date":          cp.date,
            "From Regime":   cp.from_regime,
            "To Regime":     cp.to_regime,
            "From Duration": cp.from_duration,
            "To Duration":   cp.to_duration,
            "Confidence":    cp.confidence,
            "Is Major":      cp.is_major,
        }
        for cp in changepoints
    ]
    return pd.DataFrame(rows)


def print_changepoints(changepoints: List[Changepoint], title: str = "CHANGEPOINTS") -> None:
    """Print a formatted changepoint table."""
    df = changepoints_to_dataframe(changepoints)
    if df.empty:
        print(f"\n{title}: none detected.\n")
        return
    print(f"\n{'═'*70}")
    print(title)
    print(f"{'═'*70}")
    for _, row in df.iterrows():
        marker = "★ MAJOR" if row["Is Major"] else "      "
        print(f"  {marker}  {str(row['Date'].date())}  "
              f"{row['From Regime']:30s} → {row['To Regime']:30s}  "
              f"conf={row['Confidence']:.2f}  "
              f"dur=[{row['From Duration']}, {row['To Duration']}]")
    print()


def save_changepoints(changepoints: List[Changepoint], split: str = "full") -> None:
    """Save changepoints DataFrame to CSV."""
    df = changepoints_to_dataframe(changepoints)
    path = REPORTS_DIR / f"changepoints_{split}.csv"
    df.to_csv(path, index=False)
    print(f"Changepoints saved → {path}")
