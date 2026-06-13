"""
Visualisation module for Market Regime Detection.

Generates 7 figures:
1. Price history with regime colouring
2. Hidden state timeline
3. Transition matrix heatmap
4. Regime duration distribution
5. Changepoint timeline
6. Posterior state probability plot
7. Regime occupancy statistics
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates  as mdates
import seaborn as sns

from config import FIGURES_DIR, REGIME_COLORS
from changepoints import Changepoint

# Global style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "figure.titlesize": 14,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "legend.framealpha": 0.4,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
})

_ALPHA_BAND = 0.35


def _color(regime: str) -> str:
    return REGIME_COLORS.get(regime, "#888888")


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Price history with regime colouring
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_regimes(
    prices:    pd.Series,
    labels:    pd.Series,
    title:     str = "SPY Price History with Regime Colouring",
    filename:  str = "01_price_regimes.png",
) -> None:
    """Shade price chart background by regime."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(title, y=1.01)

    # Plot price
    ax.plot(prices.index, prices.values, color="#58a6ff", linewidth=0.8, zorder=3)

    # Shade regions
    regimes = sorted(labels.unique())
    prev_lbl = labels.iloc[0]
    seg_start = labels.index[0]

    def _shade(start, end, lbl):
        ax.axvspan(start, end, color=_color(lbl), alpha=_alpha(lbl), zorder=1)

    def _alpha(lbl):
        return 0.45 if lbl == "High Volatility / Crisis" else _ALPHA_BAND

    for ts, lbl in zip(labels.index[1:], labels.iloc[1:]):
        if lbl != prev_lbl:
            _shade(seg_start, ts, prev_lbl)
            seg_start = ts
            prev_lbl  = lbl
    _shade(seg_start, labels.index[-1], prev_lbl)

    # Legend
    patches = [
        mpatches.Patch(color=_color(r), alpha=0.7, label=r)
        for r in [
            "Trending Bull",
            "Trending Bear",
            "Ranging / Sideways",
            "High Volatility / Crisis",
        ]
    ]
    ax.legend(handles=patches, loc="upper left")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="y")
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Hidden state timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_state_timeline(
    labels:   pd.Series,
    title:    str = "Hidden State Timeline",
    filename: str = "02_state_timeline.png",
) -> None:
    """Stacked bar chart showing regime occupancy over time (annual buckets)."""
    df = pd.DataFrame({"regime": labels})
    df["year"] = df.index.year

    pivot = (
        df.groupby(["year", "regime"])
          .size()
          .unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(title)

    bottom = np.zeros(len(pivot_pct))
    for regime in [
        "Trending Bull", "Ranging / Sideways",
        "Trending Bear", "High Volatility / Crisis",
    ]:
        if regime not in pivot_pct.columns:
            continue
        vals = pivot_pct[regime].values
        ax.bar(
            pivot_pct.index.astype(str), vals,
            bottom=bottom, color=_color(regime),
            alpha=0.85, label=regime, width=0.8,
        )
        bottom += vals

    ax.set_ylabel("Fraction of Year")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_xlabel("Year")
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Transition matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_transition_heatmap(
    trans_matrix: pd.DataFrame,
    title:        str = "Regime Transition Matrix",
    filename:     str = "03_transition_heatmap.png",
) -> None:
    """Annotated heatmap of empirical transition probabilities."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title)

    sns.heatmap(
        trans_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="#21262d",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        vmin=0, vmax=1,
    )
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Regime duration distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_duration_distribution(
    labels:   pd.Series,
    title:    str = "Regime Duration Distribution",
    filename: str = "04_duration_distribution.png",
) -> None:
    """Box + strip plot of consecutive run lengths per regime."""
    durations: list[dict] = []
    count = 0
    prev  = None

    for lbl in labels:
        if lbl == prev:
            count += 1
        else:
            if prev is not None:
                durations.append({"Regime": prev, "Duration": count})
            prev  = lbl
            count = 1
    if prev is not None:
        durations.append({"Regime": prev, "Duration": count})

    df = pd.DataFrame(durations)

    order = [
        "Trending Bull", "Trending Bear",
        "Ranging / Sideways", "High Volatility / Crisis",
    ]
    order = [r for r in order if r in df["Regime"].unique()]
    palette = {r: _color(r) for r in order}

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title)

    sns.boxplot(
        data=df, x="Regime", y="Duration",
        order=order, palette=palette,
        width=0.5, fliersize=3, ax=ax,
        linewidth=1.2,
    )
    ax.set_ylabel("Duration (trading days)")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    ax.grid(True, axis="y")
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Changepoint timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_changepoint_timeline(
    prices:      pd.Series,
    labels:      pd.Series,
    changepoints: List[Changepoint],
    title:       str = "Changepoint Timeline",
    filename:    str = "05_changepoint_timeline.png",
) -> None:
    """Price chart annotated with major changepoints."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(title)

    ax.plot(prices.index, prices.values, color="#58a6ff", linewidth=0.8, zorder=2)

    major = [cp for cp in changepoints if cp.is_major]
    colors_used: set[str] = set()

    for cp in major:
        col = _color(cp.to_regime)
        ax.axvline(cp.date, color=col, linewidth=1.5, linestyle="--", alpha=0.9, zorder=3)
        ax.text(
            cp.date, ax.get_ylim()[1] * 0.98,
            f"→{cp.to_regime[:4]}",
            rotation=90, va="top", ha="right",
            fontsize=6.5, color=col, alpha=0.85,
        )
        colors_used.add(cp.to_regime)

    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="y")
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Posterior state probability
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior_probabilities(
    posteriors: np.ndarray,
    dates:      pd.DatetimeIndex,
    state_map:  Dict[int, str],
    title:      str = "Posterior State Probabilities",
    filename:   str = "06_posterior_probabilities.png",
) -> None:
    """Stacked area chart of posterior state probabilities over time."""
    n_states = posteriors.shape[1]
    order = ["Trending Bull", "Trending Bear",
             "Ranging / Sideways", "High Volatility / Crisis"]

    # Map state index to name
    idx_to_name = {i: state_map[i] for i in range(n_states)}

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(title)

    bottom = np.zeros(len(dates))
    for i in range(n_states):
        name  = idx_to_name[i]
        col   = _color(name)
        probs = posteriors[:, i]
        ax.fill_between(dates, bottom, bottom + probs,
                         color=col, alpha=0.75, label=name)
        bottom += probs

    ax.set_ylabel("Posterior Probability")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Regime occupancy statistics
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_occupancy(
    train_labels: pd.Series,
    test_labels:  pd.Series,
    title:        str = "Regime Occupancy: Train vs Test",
    filename:     str = "07_regime_occupancy.png",
) -> None:
    """Grouped bar chart comparing regime frequency between splits."""
    train_dist = train_labels.value_counts(normalize=True)
    test_dist  = test_labels.value_counts(normalize=True)

    regimes = [
        "Trending Bull", "Trending Bear",
        "Ranging / Sideways", "High Volatility / Crisis",
    ]
    regimes = [r for r in regimes if r in train_dist.index or r in test_dist.index]

    x      = np.arange(len(regimes))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title)

    train_vals = [train_dist.get(r, 0.0) for r in regimes]
    test_vals  = [test_dist.get(r, 0.0)  for r in regimes]

    bars1 = ax.bar(x - width/2, train_vals, width,
                   color=[_color(r) for r in regimes],
                   alpha=0.9, label="Train")
    bars2 = ax.bar(x + width/2, test_vals, width,
                   color=[_color(r) for r in regimes],
                   alpha=0.5, label="Test", edgecolor="white", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=15, ha="right")
    ax.set_ylabel("Frequency (proportion)")
    ax.legend()
    ax.grid(True, axis="y")
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: render all figures
# ─────────────────────────────────────────────────────────────────────────────

def render_all(
    prices_full:      pd.Series,
    labels_full:      pd.Series,
    labels_train:     pd.Series,
    labels_test:      pd.Series,
    posteriors_full:  np.ndarray,
    dates_full:       pd.DatetimeIndex,
    state_map:        Dict[int, str],
    changepoints:     List[Changepoint],
    trans_matrix:     pd.DataFrame,
) -> None:
    """Generate all seven figures."""
    print("\n── Generating figures ─────────────────────────────────────────")
    plot_price_regimes(prices_full, labels_full)
    plot_state_timeline(labels_full)
    plot_transition_heatmap(trans_matrix)
    plot_duration_distribution(labels_full)
    plot_changepoint_timeline(prices_full, labels_full, changepoints)
    plot_posterior_probabilities(posteriors_full, dates_full, state_map)
    plot_regime_occupancy(labels_train, labels_test)
    print("All figures saved.")
