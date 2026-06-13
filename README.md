# Market Regime Detection using Hidden Markov Models

Identify latent market regimes in the S&P 500 ETF (SPY) using a 4-state
Gaussian HMM, with full feature engineering, temporal evaluation, changepoint
detection, and automated reporting.

---

## Project Structure

```
market_regime/
│
├── main.py               — End-to-end pipeline orchestrator
│
├── src/
│   ├── config.py         — All configuration constants and paths
│   ├── data.py           — Data download and feature engineering
│   ├── model.py          — HMM training, labeling, and inference
│   ├── profiling.py      — Per-regime performance statistics
│   ├── changepoints.py   — Changepoint detection and filtering
│   ├── evaluation.py     — Out-of-sample evaluation framework
│   ├── visualization.py  — 7 output charts
│   └── report.py         — Final analysis report generator
│   
├── outputs/
│   ├── figures/          — 7 PNG charts
│   ├── reports/          — CSV + TXT analysis reports
│   └── data/             — Pickled model, label CSVs
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires **Python 3.11+**.

---

## Run

```bash
cd market_regime
python src/main.py
```

Outputs are saved automatically to `outputs/`.

---

## Four Discovered Regimes

| Regime | Characteristics |
|---|---|
| **Trending Bull** | Positive drift, moderate volatility, sustained uptrend |
| **Trending Bear** | Negative drift, moderate volatility, sustained downtrend |
| **Ranging / Sideways** | Near-zero drift, mean-reverting, low directional momentum |
| **High Volatility / Crisis** | Elevated volatility, large swings, market stress |

State labels are assigned **data-driven** (no hardcoding) based on the
empirical mean return and volatility of each inferred HMM state.

---

## Evaluation Philosophy

This project deliberately avoids classification accuracy metrics.
The "true" regime of any market day is latent and unknowable — there is no
ground-truth label to compare against. Instead, evaluation uses:

- **Per-sample log-likelihood** (how well the model scores unseen data)
- **Regime distribution stability** (total variation distance train vs test)
- **Transition consistency** (are common transitions stable across splits?)
- **Regime persistence** (average run length, train vs test)
- **Changepoint alignment** with known market events (qualitative check)

---

## Key Configuration (src/config.py)

| Parameter              | Default    | Description                                    |
| ---------------------- | ---------- | ---------------------------------------------- |
| `N_COMPONENTS`         | 4          | Number of HMM hidden states                    |
| `TICKER`               | `SPY`      | Ticker symbol for the data                     |
| `TRAIN_END_DATE`       | 2020-12-31 | Temporal split boundary                        |
| `ROLLING_VOL_WINDOW`   | 20         | Days for volatility/momentum/autocorr          |
| `MIN_REGIME_DURATION`  | 10         | Minimum days for a valid regime run            |
| `CONFIDENCE_THRESHOLD` | 0.70       | Min posterior confidence for major changepoint |
| `RANDOM_SEED`          | 42         | Reproducibility seed                           |
