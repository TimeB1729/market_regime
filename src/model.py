"""
Hidden Markov Model training, state labeling, and inference.

Design notes
------------
* Uses hmmlearn.hmm.GaussianHMM with full covariance matrices.
* State labels are assigned *data-driven* by ranking discovered states along two
  orthogonal axes:

    axis-1 (trend):      mean log-return  → positive ⟹ Bull; negative ⟹ Bear
    axis-2 (volatility): mean rolling-vol → high ⟹ Crisis; low ⟹ Bull/Bear/Sideways

  Assignment logic:
    1. Highest volatility state            → "High Volatility / Crisis"
    2. Among remaining states:
       a. Highest mean return              → "Trending Bull"
       b. Lowest mean return               → "Trending Bear"
       c. Remaining (weakest trend/vol)    → "Ranging / Sideways"

  This guarantees exactly one label per state without any hard-coded index.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import (
    N_COMPONENTS,
    RANDOM_SEED,
    N_ITER,
    COV_TYPE,
    REGIME_LABELS,
    DATA_DIR,
)


# ── Model training ────────────────────────────────────────────────────────────

def train_hmm(X_train: pd.DataFrame) -> GaussianHMM:
    """Fit a Gaussian HMM on *X_train*.

    Parameters
    ----------
    X_train : pd.DataFrame
        Standardised feature matrix (rows = trading days).

    Returns
    -------
    GaussianHMM
        Fitted model.
    """
    model = GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type=COV_TYPE,
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    model.fit(X_train.values)
    train_ll = model.score(X_train.values)
    print(f"HMM fitted — train log-likelihood per sample: {train_ll:.4f}")
    return model


# ── State labeling ────────────────────────────────────────────────────────────

def _state_profile(
    model: GaussianHMM,
    X: pd.DataFrame,
    states: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute per-state descriptive statistics from raw (un-standardised) features.

    For labeling we need:
    * mean of log_return feature (index 0 in feature matrix)
    * mean of rolling_vol  feature (index 1 in feature matrix)

    Returns
    -------
    pd.DataFrame  indexed by state (0 … N-1)
    """
    records = []
    for s in range(model.n_components):
        mask = states == s
        subset = X.loc[mask]
        records.append({
            "state":       s,
            "mean_return": subset["log_return"].mean(),
            "mean_vol":    subset["rolling_vol"].mean(),
            "count":       mask.sum(),
        })
    return pd.DataFrame(records).set_index("state")


def assign_regime_labels(
    model: GaussianHMM,
    X_train: pd.DataFrame,
    train_states: np.ndarray,
) -> Dict[int, str]:
    """Map latent HMM state integers to economic regime strings.

    Algorithm
    ---------
    1. State with highest mean rolling-vol → "High Volatility / Crisis"
    2. Among the remaining 3 states:
       - Highest mean log-return  → "Trending Bull"
       - Lowest  mean log-return  → "Trending Bear"
       - Remainder                → "Ranging / Sideways"

    Parameters
    ----------
    model        : fitted GaussianHMM
    X_train      : original (standardised) training DataFrame
    train_states : array of inferred states on training set

    Returns
    -------
    Dict[int, str]  mapping HMM state index → regime label string
    """
    profile = _state_profile(model, X_train, train_states, list(X_train.columns))

    # Step 1 — identify crisis state (highest volatility)
    crisis_state = int(profile["mean_vol"].idxmax())

    remaining = profile.drop(index=crisis_state)

    # Step 2 — bull = highest return, bear = lowest return, sideways = middle
    bull_state     = int(remaining["mean_return"].idxmax())
    bear_state     = int(remaining["mean_return"].idxmin())
    sideways_state = int(remaining.drop(index=[bull_state, bear_state]).index[0])

    mapping = {
        bull_state:     "Trending Bull",
        bear_state:     "Trending Bear",
        sideways_state: "Ranging / Sideways",
        crisis_state:   "High Volatility / Crisis",
    }

    print("\nData-driven state → regime mapping:")
    for state, label in sorted(mapping.items()):
        row = profile.loc[state]
        print(f"  State {state}: {label:30s} "
              f"mean_return={row['mean_return']:+.5f}  "
              f"mean_vol={row['mean_vol']:.5f}  "
              f"n={int(row['count'])}")
    return mapping


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_states(
    model: GaussianHMM,
    X: pd.DataFrame,
    state_map: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Run Viterbi decoding and compute posterior probabilities.

    Parameters
    ----------
    model     : fitted GaussianHMM
    X         : feature matrix (train or test)
    state_map : Dict[int, str] from assign_regime_labels

    Returns
    -------
    states     : np.ndarray  integer state sequence
    posteriors : np.ndarray  (n_obs × n_components) posterior probabilities
    labels     : pd.Series   regime label strings, indexed like X
    """
    states     = model.predict(X.values)
    posteriors = model.predict_proba(X.values)
    labels     = pd.Series(
        [state_map[s] for s in states],
        index=X.index,
        name="regime",
    )
    return states, posteriors, labels


def compute_log_likelihood(model: GaussianHMM, X: pd.DataFrame) -> float:
    """Return per-sample log-likelihood of *X* under *model*."""
    return model.score(X.values) / len(X)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: GaussianHMM, path: Path | None = None) -> Path:
    """Pickle the fitted HMM model."""
    if path is None:
        path = DATA_DIR / "hmm_model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"Model saved → {path}")
    return path


def load_model(path: Path | None = None) -> GaussianHMM:
    """Load a pickled HMM model."""
    if path is None:
        path = DATA_DIR / "hmm_model.pkl"
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    return model
