"""
Analysis report generator.

Produces a human-readable text report answering the six analysis questions:
1.  What regimes were discovered?
2.  How persistent are they?
3.  Which transitions occur most frequently?
4.  Do major changepoints align with known market events?
5.  Does the HMM identify meaningful market structure?
6.  Can regime information plausibly be used for risk management / strategy switching?
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import REPORTS_DIR
from changepoints import Changepoint


# ── Known market events (for qualitative alignment check) ────────────────────

KNOWN_EVENTS: list[tuple[str, str, str]] = [
    ("2010-04-01", "2010-07-01", "Flash Crash (May 2010)"),
    ("2011-07-01", "2011-10-01", "US Debt Ceiling / Euro Crisis"),
    ("2015-07-01", "2016-02-01", "China/Oil Selloff 2015-16"),
    ("2018-09-01", "2019-01-01", "Q4 2018 Selloff"),
    ("2020-02-01", "2020-04-15", "COVID-19 Crash"),
    ("2022-01-01", "2022-10-01", "Rate Hike Bear Market 2022"),
    ("2023-03-01", "2023-04-01", "SVB Banking Crisis"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _top_transitions(trans_matrix: pd.DataFrame, n: int = 5) -> list[str]:
    """Return the *n* most frequent off-diagonal transitions as strings."""
    # Flatten and remove self-transitions
    flat = []
    for fr in trans_matrix.index:
        for to in trans_matrix.columns:
            if fr != to:
                flat.append((trans_matrix.loc[fr, to], fr, to))
    flat.sort(reverse=True)
    return [f"{fr} → {to}  ({p:.3f})" for p, fr, to in flat[:n]]


def _event_alignment(changepoints: List[Changepoint]) -> list[str]:
    """Check whether major changepoints fall near known market events."""
    lines = []
    for cp in changepoints:
        if not cp.is_major:
            continue
        for start, end, desc in KNOWN_EVENTS:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            if s <= cp.date <= e:
                lines.append(
                    f"  ✓  {cp.date.date()}  [{cp.from_regime} → {cp.to_regime}]"
                    f"  aligns with: {desc}"
                )
                break
        else:
            lines.append(
                f"  –  {cp.date.date()}  [{cp.from_regime} → {cp.to_regime}]"
                f"  (no direct match in known-event list)"
            )
    return lines


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(
    profile_train:  pd.DataFrame,
    profile_test:   pd.DataFrame,
    persist_train:  pd.DataFrame,
    persist_test:   pd.DataFrame,
    trans_train:    pd.DataFrame,
    trans_test:     pd.DataFrame,
    changepoints:   List[Changepoint],
    likelihood:     Dict[str, float],
    dist_stability: pd.DataFrame,
) -> str:
    """Compile the full analysis report as a string."""

    sep = "\n" + "─" * 70 + "\n"
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append(" MARKET REGIME DETECTION — ANALYSIS REPORT")
    lines.append("=" * 70)

    # Q1 — Regimes discovered
    lines.append(sep + "Q1: WHAT REGIMES WERE DISCOVERED?\n")
    for regime, row in profile_train.iterrows():
        lines.append(
            f"  ► {regime}\n"
            f"    Mean daily return  : {row['Mean Return (daily)']:+.5f}\n"
            f"    Daily volatility   :  {row['Volatility (daily)']:.5f}\n"
            f"    Ann. Sharpe        :  {row['Ann. Sharpe']:.3f}\n"
            f"    Max Drawdown       :  {row['Max Drawdown']*100:.2f}%\n"
            f"    Frequency (train)  :  {row['Frequency (%)']:.2f}%\n"
        )

    # Q2 — Persistence
    lines.append(sep + "Q2: HOW PERSISTENT ARE THEY?\n")
    lines.append("Train persistence:")
    lines.append(persist_train.round(1).to_string())
    lines.append("\nTest persistence:")
    lines.append(persist_test.round(1).to_string())

    # Q3 — Most frequent transitions
    lines.append(sep + "Q3: WHICH TRANSITIONS OCCUR MOST FREQUENTLY?\n")
    lines.append("Top transitions (train):")
    for t in _top_transitions(trans_train):
        lines.append(f"  {t}")
    lines.append("\nTop transitions (test):")
    for t in _top_transitions(trans_test):
        lines.append(f"  {t}")

    # Q4 — Changepoint alignment
    lines.append(sep + "Q4: DO MAJOR CHANGEPOINTS ALIGN WITH KNOWN MARKET EVENTS?\n")
    alignment = _event_alignment(changepoints)
    if alignment:
        lines.extend(alignment)
    else:
        lines.append("  No major changepoints detected.")

    # Q5 — Meaningful structure
    lines.append(sep + "Q5: DOES THE HMM IDENTIFY MEANINGFUL MARKET STRUCTURE?\n")
    train_ll = likelihood["train_ll"]
    test_ll  = likelihood["test_ll"]
    ll_ratio = likelihood["ll_ratio"]
    tvd = dist_stability["abs_diff"].sum() / 2

    lines.append(
        f"  Train log-likelihood (per sample): {train_ll:.5f}\n"
        f"  Test  log-likelihood (per sample): {test_ll:.5f}\n"
        f"  LL ratio (test/train)            : {ll_ratio:.4f}\n"
        f"  Total variation distance (dist.) : {tvd:.4f}\n\n"
        "  Interpretation:\n"
        "  • A Sharpe ratio > 1.0 in Trending Bull and near 0 / negative in\n"
        "    Trending Bear confirms trend-directional regime structure.\n"
        "  • High Volatility / Crisis should show the largest annualised\n"
        "    volatility and deepest drawdown, consistent with crisis labeling.\n"
        "  • LL ratio close to 1.0 indicates the model generalises well\n"
        "    out-of-sample — the latent structure is stable across time.\n"
        "  • Low TVD (< 0.15) indicates regime frequency stability between\n"
        "    training and test periods.\n"
        "  • Changepoints coinciding with COVID-19 (Feb/Mar 2020),\n"
        "    2015–16 China oil selloff, and 2022 bear market confirm\n"
        "    economic interpretability of the discovered structure."
    )

    # Q6 — Risk management / strategy switching
    lines.append(sep + "Q6: CAN REGIME INFORMATION BE USED FOR RISK MANAGEMENT?\n")
    lines.append(
        "  YES — with appropriate caveats:\n\n"
        "  ► Strategy switching:\n"
        "    • Trending Bull    → run momentum / trend-following strategies.\n"
        "    • Trending Bear    → reduce equity exposure; rotate to bonds/cash.\n"
        "    • Ranging/Sideways → mean-reversion strategies (pairs, stat-arb).\n"
        "    • Crisis           → volatility-targeting, safe-haven allocation.\n\n"
        "  ► Risk sizing:\n"
        "    Annualised volatility per regime can feed directly into a\n"
        "    volatility-targeting framework (e.g., target 10% ann. vol):\n"
        "    position_size = target_vol / regime_ann_vol.\n\n"
        "  ► Caveats:\n"
        "    • Regimes are inferred with posterior uncertainty — always\n"
        "      weight decisions by posterior probability, not point estimates.\n"
        "    • Regime transitions are detected with a lag (minimum duration\n"
        "      filter); act on confirmed transitions, not single-day signals.\n"
        "    • The HMM is a generative model trained on historical data;\n"
        "      future structural breaks (new regimes) may not be captured.\n"
        "    • This output is research / educational — not financial advice."
    )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def save_report(report_text: str) -> Path:
    """Write the report string to a text file."""
    path = REPORTS_DIR / "analysis_report.txt"
    path.write_text(report_text, encoding="utf-8")
    print(f"Analysis report saved → {path}")
    return path
