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

    pip install -r requirements.txt

Requires **Python 3.11+**.

---

## Run

    cd market_regime
    python src/main.py

Outputs are saved automatically to `outputs/`.

---

## Four Discovered Regimes

| Regime | Characteristics |
|---|---|
| **Trending Bull** | Positive drift, moderate volatility, sustained uptrend |
| **Trending Bear** | Negative drift, moderate volatility, sustained downtrend |
| **Ranging / Sideways** | Near-zero drift, mean-reverting, low directional momentum |
| **High Volatility / Crisis** | Elevated volatility, large swings, market stress |

State labels are assigned **data-driven** (no hardcoding) based on the empirical mean return and volatility of each inferred HMM state.

---

## Theory: Regime-Aware Signal Aggregation Framework

### Motivation
The purpose of market regime detection is not merely to classify historical market behavior. In practical quantitative trading systems, regime information acts as a contextual layer that determines how other trading signals should be interpreted. Many signals that perform well in one market environment may fail in another. 

| Signal Type | Typical Strength |
| :--- | :--- |
| **Momentum** | Trending markets |
| **Mean Reversion** | Sideways markets |
| **Volatility Trading** | Crisis markets |
| **Carry / Risk Premia** | Stable bull markets |

Consequently, a regime detector can be viewed as a market-state estimator rather than a standalone trading strategy.

### Regime Probabilities
The Hidden Markov Model produces posterior state probabilities:

    $$ \mathrm{P}(S_t = k | X_t) $$

where:
* **$S_t$** is the hidden regime at time $t$
* **$X_t$** represents observed market features
* **$k$** is an element of {1, 2, 3, 4}

Unlike a hard classification, these probabilities express uncertainty regarding the current market state. 

**Example:** 

| Regime            | Probability |
| :---------------- | :---------- |
| **Trending Bull** | 0.62        |
| **Trending Bear** | 0.08        |
| **Sideways**      | 0.21        |
| **Crisis**        | 0.09        |

This probabilistic representation is often more useful than relying on a single, rigid regime label.

### Regime-Specific Signals
Suppose we define **$M_t$** as a momentum signal, **$R_t$** as a mean-reversion signal, and **$V_t$** as a volatility signal. Each signal may have different predictive power under different regimes:

| Regime       | Momentum Weight | Mean Reversion Weight | Volatility Weight |
| :----------- | :-------------- | :-------------------- | :---------------- |
| **Bull**     | High            | Low                   | Low               |
| **Bear**     | Medium          | Low                   | Medium            |
| **Sideways** | Low             | High                  | Low               |
| **Crisis**   | Low             | Low                   | High              |

### Mixture-of-Signals Framework
Let **$f_k(X_t)$** represent the trading signal associated with regime $k$. The aggregate signal becomes:

    $$ \text{Signal}_t = \sum_{k=1}^{K} \left[ \mathrm{P}(S_t = k | X_t) \times f_k(X_t) \right] $$

**Interpretation:**
* Each regime contributes its own specialized signal.
* Contributions are weighted by the probability that the regime is currently active.
* Regime uncertainty is incorporated naturally, creating a dynamic, regime-aware signal ensemble.

### Strategy Switching
An alternative implementation uses discrete strategy selection, where the regime detector acts as a routing mechanism deciding which strategy should currently be active:

| Regime | Active Strategy |
| :--- | :--- |
| **Trending Bull** | Trend Following |
| **Trending Bear** | Defensive Allocation |
| **Sideways** | Mean Reversion |
| **Crisis** | Risk Reduction / Volatility Targeting |

### Risk Management Applications
Regime information may be used to dynamically adjust exposure. Let **$\sigma_k$** be the annualized volatility associated with regime k. Position size may then be scaled according to:
    $$ \text{PositionSize} = \frac{\text{TargetVolatility}}{\sigma_k} $$

This formulation automatically reduces leverage during crisis regimes and increases exposure during stable regimes.

### Regime Forecasting
The transition matrix of the HMM provides **$\mathrm{P}(S_{t+1} | S_t)$** and, more generally, **$\mathrm{P}(S_{t+h} | S_t)$** for future horizons $h$. This allows estimation of forward-looking quantities such as:
* The probability of entering a crisis regime within 5 trading days.
* The probability that a bull regime persists for another month.
* The probability of transition from sideways to trending conditions.

Such forecasts are often more actionable than predicting future returns directly.

### Research Direction
The current project focuses on detecting historical and current market regimes. Future extensions may include:
* Regime-specific alpha models.
* Regime-weighted signal ensembles.
* Dynamic portfolio allocation and Volatility targeting.
* Regime forecasting using transition probabilities.
* Hidden Semi-Markov Models for duration-aware regimes.
* Bayesian regime-switching models.

The long-term objective is not merely to identify market states, but to build adaptive trading and risk-management systems whose behavior fundamentally changes according to the inferred market environment.

---

## Evaluation Philosophy

This project deliberately avoids classification accuracy metrics. The "true" regime of any market day is latent and unknowable — there is no ground-truth label to compare against. Instead, evaluation uses:

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