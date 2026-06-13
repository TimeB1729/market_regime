"""
main.py — End-to-end pipeline for Market Regime Detection using HMMs.

Usage
-----
    cd market_regime
    python src/main.py

Outputs are written to:
    outputs/figures/   — 7 PNG charts
    outputs/reports/   — CSV and TXT reports
    outputs/data/      — pickled model and label CSVs
"""

from __future__ import annotations

import sys
import os

# Ensure src/ is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from src.config import (
    TICKER, START_DATE, TRAIN_END_DATE,
    MIN_REGIME_DURATION, RANDOM_SEED, DATA_DIR,
)
from src.data import download_data, build_features, temporal_split
from src.model import (
    train_hmm, assign_regime_labels, infer_states,
    compute_log_likelihood, save_model,
)
from src.profiling import profile_regimes, print_regime_report, save_regime_report
from src.changepoints import (
    detect_changepoints, filter_major_changepoints,
    print_changepoints, save_changepoints,
)
from src.evaluation import (
    evaluate_likelihood, evaluate_distribution_stability,
    evaluate_transition_consistency, evaluate_persistence,
    save_evaluation,
)
from src.visualization import render_all
from src.report import build_report, save_report

# ── Transition matrix helper ──────────────────────────────────────────────────

def _empirical_transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Build a normalised empirical transition matrix from label series."""
    regimes = sorted(labels.unique())
    mat = pd.DataFrame(0.0, index=regimes, columns=regimes)
    prev = None
    for lbl in labels:
        if prev is not None:
            mat.loc[prev, lbl] += 1
        prev = lbl
    row_sums = mat.sum(axis=1).replace(0, 1)
    return mat.div(row_sums, axis=0)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Execute the full Market Regime Detection pipeline."""

    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 70)
    print(" MARKET REGIME DETECTION — HMM PIPELINE")
    print("=" * 70)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/7] Downloading and engineering features …")
    raw = download_data(TICKER, start=START_DATE)
    features, prices = build_features(raw)
    X_train, X_test, p_train, p_test = temporal_split(
        features, prices, TRAIN_END_DATE
    )

    # ── 2. Model training ─────────────────────────────────────────────────────
    print("\n[2/7] Training HMM …")
    model = train_hmm(X_train)
    save_model(model)

    # ── 3. State inference & labeling ─────────────────────────────────────────
    print("\n[3/7] Inferring states and labeling regimes …")

    # Training set
    train_states, train_post, train_labels = infer_states(model, X_train, {i: str(i) for i in range(model.n_components)})
    state_map = assign_regime_labels(model, X_train, train_states)

    # Re-infer with final state map
    train_states, train_post, train_labels = infer_states(model, X_train, state_map)
    test_states,  test_post,  test_labels  = infer_states(model, X_test,  state_map)

    # Full set
    states_full, post_full, labels_full = infer_states(
        model, features, state_map
    )

    # Save labels
    labels_full.to_csv(DATA_DIR / "regime_labels_full.csv", header=True)
    train_labels.to_csv(DATA_DIR / "regime_labels_train.csv", header=True)
    test_labels.to_csv(DATA_DIR  / "regime_labels_test.csv",  header=True)

    # ── 4. Regime profiling ───────────────────────────────────────────────────
    print("\n[4/7] Profiling regimes …")
    profile_train = profile_regimes(train_labels, p_train)
    profile_test  = profile_regimes(test_labels,  p_test)
    profile_full  = profile_regimes(labels_full,  prices)

    print_regime_report(profile_train)
    save_regime_report(profile_train, "train")
    save_regime_report(profile_test,  "test")
    save_regime_report(profile_full,  "full")

    # ── 5. Changepoint detection ──────────────────────────────────────────────
    print("\n[5/7] Detecting changepoints …")
    all_cps   = detect_changepoints(labels_full, post_full, MIN_REGIME_DURATION)
    major_cps = filter_major_changepoints(all_cps, labels_full)

    print(f"  Total changepoints  : {len(all_cps)}")
    print(f"  Major changepoints  : {len(major_cps)}")
    print_changepoints(major_cps, title="MAJOR CHANGEPOINTS")

    save_changepoints(all_cps,   "all")
    save_changepoints(major_cps, "major")

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print("\n[6/7] Evaluating model …")
    likelihood = evaluate_likelihood(model, X_train, X_test)
    dist_stab  = evaluate_distribution_stability(train_labels, test_labels)
    trans_train, trans_test = evaluate_transition_consistency(train_labels, test_labels)
    persist_train, persist_test = evaluate_persistence(train_labels, test_labels)

    save_evaluation(
        likelihood, dist_stab,
        trans_train, trans_test,
        persist_train, persist_test,
    )

    # ── 7. Visualisations ─────────────────────────────────────────────────────
    print("\n[7/7] Generating visualisations …")
    render_all(
        prices_full     = prices,
        labels_full     = labels_full,
        labels_train    = train_labels,
        labels_test     = test_labels,
        posteriors_full = post_full,
        dates_full      = features.index,
        state_map       = state_map,
        changepoints    = major_cps,
        trans_matrix    = trans_train,
    )

    # ── Final report ──────────────────────────────────────────────────────────
    report_text = build_report(
        profile_train  = profile_train,
        profile_test   = profile_test,
        persist_train  = persist_train,
        persist_test   = persist_test,
        trans_train    = trans_train,
        trans_test     = trans_test,
        changepoints   = major_cps,
        likelihood     = likelihood,
        dist_stability = dist_stab,
    )
    print(report_text)
    save_report(report_text)

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print(f" Figures  → {str((DATA_DIR / '..').resolve() / 'figures')}")
    print(f" Reports  → {str((DATA_DIR / '..').resolve() / 'reports')}")
    print("=" * 70 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
