"""
Out-of-sample evaluation for the HMM regime model.

Why classification accuracy is NOT appropriate
-----------------------------------------------
Hidden Markov Models model latent (unobserved) states.  The "true" regime of
any market day is fundamentally unknowable — there is no ground-truth label to
compare against.  Even if a human annotated a dataset, those labels would be
subjective and retrospective.  Standard classification metrics (accuracy,
precision, recall, F1) assume access to hard, known ground-truth labels.
Applying them here would:

  1. Require inventing labels, which would be circular (we'd be testing how
     well the model reproduces our own prior beliefs).
  2. Ignore the probabilistic, sequential nature of the HMM.
  3. Ignore regime persistence and transition dynamics, which carry the
     economically meaningful signal.

Instead, we evaluate via:
  * Per-sample log-likelihood (measures how well the model assigns probability
    mass to unseen observations).
  * Regime distribution stability (do out-of-sample regime frequencies look
    similar to in-sample?).
  * Transition consistency (are the most common transitions the same
    in-sample and out-of-sample?).
  * Regime persistence (average run length in test set vs training set).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import REPORTS_DIR


# ── Likelihood metrics ────────────────────────────────────────────────────────

def evaluate_likelihood(
    model:   GaussianHMM,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> Dict[str, float]:
    """Compute and compare per-sample log-likelihoods.

    Parameters
    ----------
    model   : fitted GaussianHMM
    X_train : training feature matrix
    X_test  : test feature matrix

    Returns
    -------
    dict with keys 'train_ll', 'test_ll', 'll_ratio'
    """
    train_ll = model.score(X_train.values) / len(X_train)
    test_ll  = model.score(X_test.values)  / len(X_test)
    ll_ratio = test_ll / train_ll  # close to 1.0 = stable model

    print("\n── Likelihood Evaluation ──────────────────────────────────────")
    print(f"  Train log-likelihood (per sample): {train_ll:.6f}")
    print(f"  Test  log-likelihood (per sample): {test_ll:.6f}")
    print(f"  Ratio (test / train)             : {ll_ratio:.4f}")
    print("  (Ratio ≈ 1.0 indicates stable fit; < 0.80 suggests degradation)")

    return {"train_ll": train_ll, "test_ll": test_ll, "ll_ratio": ll_ratio}


# ── Distribution stability ────────────────────────────────────────────────────

def regime_distribution(labels: pd.Series) -> pd.Series:
    """Compute normalised regime frequency distribution."""
    return labels.value_counts(normalize=True).sort_index()


def evaluate_distribution_stability(
    train_labels: pd.Series,
    test_labels:  pd.Series,
) -> pd.DataFrame:
    """Compare regime frequency distributions between train and test sets.

    Parameters
    ----------
    train_labels : pd.Series
    test_labels  : pd.Series

    Returns
    -------
    pd.DataFrame  with 'train_freq', 'test_freq', 'abs_diff' columns
    """
    train_dist = regime_distribution(train_labels).rename("train_freq")
    test_dist  = regime_distribution(test_labels).rename("test_freq")

    df = pd.concat([train_dist, test_dist], axis=1).fillna(0.0)
    df["abs_diff"] = (df["train_freq"] - df["test_freq"]).abs()

    print("\n── Regime Distribution Stability ──────────────────────────────")
    print(df.to_string(float_format="{:.4f}".format))
    print(f"\n  Total variation distance: {df['abs_diff'].sum()/2:.4f}")
    print("  (TVD ≈ 0 = identical; TVD = 1 = completely different)")

    return df


# ── Transition consistency ────────────────────────────────────────────────────

def _transition_counts(labels: pd.Series) -> pd.DataFrame:
    """Build normalised regime transition matrix from a label series."""
    regimes = sorted(labels.unique())
    mat = pd.DataFrame(0.0, index=regimes, columns=regimes)

    prev = None
    for lbl in labels:
        if prev is not None:
            mat.loc[prev, lbl] += 1
        prev = lbl

    row_sums = mat.sum(axis=1)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    return mat.div(row_sums, axis=0)


def evaluate_transition_consistency(
    train_labels: pd.Series,
    test_labels:  pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare transition matrices between train and test.

    Returns
    -------
    train_trans, test_trans  (both normalised DataFrames)
    """
    train_trans = _transition_counts(train_labels)
    test_trans  = _transition_counts(test_labels)

    print("\n── Transition Consistency ─────────────────────────────────────")
    print("Train transition matrix:")
    print(train_trans.round(3).to_string())
    print("\nTest transition matrix:")
    print(test_trans.round(3).to_string())

    return train_trans, test_trans


# ── Persistence statistics ────────────────────────────────────────────────────

def persistence_stats(labels: pd.Series) -> pd.DataFrame:
    """Compute regime duration statistics.

    Returns
    -------
    pd.DataFrame  with mean/median/max duration per regime
    """
    durations: dict[str, list[int]] = {}
    count = 0
    prev  = None

    for lbl in labels:
        if lbl == prev:
            count += 1
        else:
            if prev is not None:
                durations.setdefault(prev, []).append(count)
            prev  = lbl
            count = 1

    if prev is not None:
        durations.setdefault(prev, []).append(count)

    rows = []
    for regime, durs in sorted(durations.items()):
        rows.append({
            "Regime":       regime,
            "Mean Dur":     np.mean(durs),
            "Median Dur":   np.median(durs),
            "Max Dur":      max(durs),
            "N Episodes":   len(durs),
        })

    return pd.DataFrame(rows).set_index("Regime")


def evaluate_persistence(
    train_labels: pd.Series,
    test_labels:  pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare persistence stats between train and test."""
    train_p = persistence_stats(train_labels)
    test_p  = persistence_stats(test_labels)

    print("\n── Persistence Statistics ─────────────────────────────────────")
    print("Train:")
    print(train_p.round(2).to_string())
    print("\nTest:")
    print(test_p.round(2).to_string())

    return train_p, test_p


# ── Save evaluation results ───────────────────────────────────────────────────

def save_evaluation(
    likelihood_metrics: Dict[str, float],
    dist_stability:     pd.DataFrame,
    train_trans:        pd.DataFrame,
    test_trans:         pd.DataFrame,
    train_persist:      pd.DataFrame,
    test_persist:       pd.DataFrame,
) -> None:
    """Save all evaluation artefacts to CSV."""
    pd.DataFrame([likelihood_metrics]).to_csv(
        REPORTS_DIR / "likelihood_metrics.csv", index=False
    )
    dist_stability.to_csv(REPORTS_DIR / "distribution_stability.csv")
    train_trans.to_csv(REPORTS_DIR  / "transition_matrix_train.csv")
    test_trans.to_csv(REPORTS_DIR   / "transition_matrix_test.csv")
    train_persist.to_csv(REPORTS_DIR / "persistence_train.csv")
    test_persist.to_csv(REPORTS_DIR  / "persistence_test.csv")
    print("\nEvaluation CSVs saved to:", REPORTS_DIR)
